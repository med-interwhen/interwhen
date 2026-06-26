"""VitaBench overlay file — modified from the original VitaBench repo
(https://github.com/meituan-longcat/vitabench), at src/vita/orchestrator/orchestrator.py.
Everything is verbatim from the original except for the following changes:

1. Added a ``verifier`` constructor argument plus completeness-retry state
   (``VITA_MAX_COMPLETENESS_RETRIES``) and ``verifier_feedback_history``.
2. Attached the verifier's ``soundness_log`` to the finalized ``SimulationRun``.
3. On agent stop, run the verifier's completeness check and, if it returns
   feedback, inject a synthetic ``completeness_check`` tool_call/tool_result pair
   to send the agent back instead of stopping.
4. In solo mode, nudge the agent back with a user message when it emits
   non-tool-call text.
5. Before executing each agent tool call, run the verifier's soundness check;
   on feedback, return it as an error ToolMessage and skip execution (blocking),
   and feed successful tool responses to the harness via ``observe_tool_response``.
"""
import os
import time
import uuid
from copy import deepcopy
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional, Dict

from loguru import logger

from vita.agent.base import BaseAgent, is_valid_agent_history_message
from vita.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from vita.data_model.simulation import SimulationRun, TerminationReason
from vita.data_model.tasks import Task
from vita.environment.db import DB
from vita.environment.environment import Environment
from vita.user.base import BaseUser, is_valid_user_history_message
from vita.user.user_simulator import UserSimulator, UserState
from vita.utils.llm_utils import get_cost
from vita.utils.utils import format_time, get_now, DATA_DIR
from vita.config import DEFAULT_LANGUAGE


class Role(str, Enum):
    AGENT = "agent"
    USER = "user"
    ENV = "env"


def get_default_first_agent_message(language: str = None) -> AssistantMessage:
    """Get the default first agent message based on language"""
    if language is None:
        language = DEFAULT_LANGUAGE
    
    content = "你好，请问需要什么服务？" if language == "chinese" else "Hello, how can I help you?"
    return AssistantMessage(
        role="assistant", content=content, cost=0.0
    )


class Orchestrator:
    """
    Orchestrator for the simulation given a task.
    Passes messages between the Agent, User, and Environment.
    """

    def __init__(
        self,
        domain: str,
        agent: BaseAgent,
        user: BaseUser,
        environment: Environment,
        task: Task,
        max_steps: int = 100,
        max_errors: int = 10,
        seed: Optional[int] = None,
        solo_mode: bool = False,
        language: str = None,
        verifier: Optional[Any] = None,
    ):
        self.domain = domain
        self.agent = agent
        self.user = user
        self.environment = environment
        self.task = task
        self.seed = seed
        self.solo_mode = solo_mode
        self.language = language
        self.agent_state: Optional[Any] = None
        self.user_state: Optional[UserState] = None
        self.trajectory: list[Message] = []
        self.max_steps = max_steps
        self.max_errors = max_errors
        self.step_count = 0
        self.done = False
        self.termination_reason: Optional[TerminationReason] = None
        self.num_errors = 0
        self.from_role: Optional[Role] = None
        self.to_role: Optional[Role] = None
        self.message: Optional[Message] = None
        self.verifier = verifier
        self.verifier_feedback_history: list = []
        self._completeness_retries = 0
        self._max_completeness_retries = int(os.environ.get("VITA_MAX_COMPLETENESS_RETRIES", 1))

    def initialize(self):
        """
        Initialize the orchestrator.
        - If the tasks specifies an initial state, use it to initialize the environment.
        - Initialize the agent and user states.
        - Send the first message (default message from the agent to the user).
        """
        message_history = (
            deepcopy(self.task.message_history)
            if self.task is not None and self.task.message_history is not None
            else []
        )
        for msg in message_history:
            msg.turn_idx = None

        message_history = self._add_timestamps(message_history)

        if self.seed is not None:
            self.agent.set_seed(self.seed)
            self.user.set_seed(self.seed)

        if len(message_history) > 0:
            self.validate_message_history(message_history)

            last_message = message_history[-1]
            if isinstance(last_message, AssistantMessage):
                self.from_role = Role.AGENT
                if not last_message.is_tool_call():
                    self.to_role = Role.USER
                else:
                    self.to_role = Role.ENV
                self.agent_state = self.agent.get_init_state(
                    message_history=[
                        msg
                        for msg in message_history
                        if is_valid_agent_history_message(msg)
                    ]
                )
                self.user_state = self.user.get_init_state(
                    message_history=[
                        msg
                        for msg in message_history[:-1]
                        if is_valid_user_history_message(msg)
                    ]
                )
                self.message = last_message
                if self.agent.is_stop(last_message):
                    self.done = True
                    self.termination_reason = TerminationReason.AGENT_STOP
            elif isinstance(last_message, UserMessage):
                self.from_role = Role.USER
                if not last_message.is_tool_call():
                    self.to_role = Role.AGENT
                else:
                    self.to_role = Role.ENV
                self.user_state = self.user.get_init_state(
                    message_history=[
                        msg
                        for msg in message_history
                        if is_valid_user_history_message(msg)
                    ]
                )
                self.agent_state = self.agent.get_init_state(
                    message_history=[
                        msg
                        for msg in message_history[:-1]
                        if is_valid_agent_history_message(msg)
                    ]
                )
                self.message = last_message
                self.done = UserSimulator.is_stop(last_message)
                if self.done:
                    self.termination_reason = TerminationReason.USER_STOP
            elif isinstance(last_message, ToolMessage):
                self.from_role = Role.ENV
                if last_message.requestor == "assistant":
                    self.to_role = Role.AGENT
                    self.agent_state = self.agent.get_init_state(
                        message_history=[
                            msg
                            for msg in message_history[:-1]
                            if is_valid_agent_history_message(msg)
                        ]
                    )
                    self.user_state = self.user.get_init_state(
                        message_history=[
                            msg
                            for msg in message_history
                            if is_valid_user_history_message(msg)
                        ]
                    )
                else:
                    self.to_role = Role.USER
                    self.agent_state = self.agent.get_init_state(
                        message_history=[
                            msg
                            for msg in message_history
                            if is_valid_agent_history_message(msg)
                        ]
                    )
                    self.user_state = self.user.get_init_state(
                        message_history=[
                            msg
                            for msg in message_history[:-1]
                            if is_valid_user_history_message(msg)
                        ]
                    )
                self.message = last_message
            else:
                raise ValueError(
                    f"Last message should be of type AssistantMessage, UserMessage, or ToolMessage, got {type(last_message)}"
                )
            self.trajectory = message_history

        else:
            self.agent_state = self.agent.get_init_state()
            self.user_state = self.user.get_init_state()
            first_message = deepcopy(get_default_first_agent_message(self.language))
            first_message.timestamp = get_now()
            self.trajectory = [first_message]
            self.message = first_message
            self.from_role = Role.AGENT
            self.to_role = Role.USER

        # with open(DATA_DIR / "tools" / f"{self.domain}.json", 'w') as f:
        #     json.dump([tools.openai_schema for tools in self.agent.tools], f, indent=4, ensure_ascii=False)

    def run(self) -> SimulationRun:
        """
        Run the simulation.

        Returns:
            SimulationRun: The simulation run.
        """
        start_time = get_now()
        start = time.perf_counter()
        self.initialize()
        while not self.done:
            self.step()
            if self.step_count >= self.max_steps:
                self.done = True
                self.termination_reason = TerminationReason.MAX_STEPS
            if self.num_errors >= self.max_errors:
                self.done = True
                self.termination_reason = TerminationReason.TOO_MANY_ERRORS
        duration = time.perf_counter() - start
        messages = self.get_trajectory()
        res = get_cost(messages)
        if res is None:
            agent_cost, user_cost = None, None
        else:
            agent_cost, user_cost = res

        simulation_run = SimulationRun(
            id=str(uuid.uuid4()),
            task_id=self.task.id,
            start_time=start_time,
            end_time=get_now(),
            duration=duration,
            termination_reason=self.termination_reason.value,
            reward_info=None,
            user_cost=user_cost,
            agent_cost=agent_cost,
            messages=messages,
            seed=self.seed,
            states=self.get_states(self.environment.tools.db, self.environment.tools.db.time),
            soundness_log=self.verifier.soundness_call_log if self.verifier and hasattr(self.verifier, "soundness_call_log") else None,
        )
        return simulation_run

    def step(self):
        """
        Perform one step of the simulation.
        Sends self.message from self.from_role to self.to_role
        This can either be a message from agent to user/environment, environment to agent, or user to agent
        Updates self.trajectory
        """
        if self.done:
            raise ValueError("Simulation is done")
        logger.debug(
            f"Step {self.step_count}. Sending message from {self.from_role} to {self.to_role}"
        )
        logger.debug(
            f"Step {self.step_count}.\nFrom role: {self.from_role}\nTo role: {self.to_role}\nMessage: {self.message}"
        )
        if self.from_role in [Role.AGENT, Role.ENV] and self.to_role == Role.USER:
            user_msg, self.user_state = self.user.generate_next_message(
                self.message, self.user_state
            )
            user_msg.validate()
            if UserSimulator.is_stop(user_msg):
                self.done = True
                self.termination_reason = TerminationReason.USER_STOP
            self.trajectory.append(user_msg)
            self.message = user_msg
            self.from_role = Role.USER
            if user_msg.is_tool_call():
                self.to_role = Role.ENV
            else:
                self.to_role = Role.AGENT
        elif (
            self.from_role == Role.USER or self.from_role == Role.ENV
        ) and self.to_role == Role.AGENT:
            # Retry up to 3 times if agent generates invalid message
            max_retries = 3
            retry_count = 0
            agent_msg = None
            original_agent_state = deepcopy(self.agent_state)  # Save original state
            
            while retry_count < max_retries:
                # Use a copy of the original state for each retry
                current_agent_state = deepcopy(original_agent_state)
                agent_msg, updated_agent_state = self.agent.generate_next_message(
                    self.message, current_agent_state
                )
                
                # Check if the message is valid (has text content or is tool call)
                if agent_msg.has_text_content() or agent_msg.is_tool_call():
                    # Only update the actual agent state if we get a valid message
                    self.agent_state = updated_agent_state
                    break
                
                retry_count += 1
                logger.warning(f"Agent generated invalid message (attempt {retry_count}/{max_retries}): {agent_msg}")
            
            # If all retries failed, terminate with INVALID_AGENT_MESSAGE
            if retry_count >= max_retries:
                self.done = True
                self.termination_reason = TerminationReason.INVALID_AGENT_MESSAGE
                return
            
            agent_msg.validate()
            completeness_feedback = None
            if self.agent.is_stop(agent_msg):
                # Run completeness check before allowing stop
                remaining = self._max_completeness_retries - self._completeness_retries
                if remaining > 0 and self.verifier:
                    states = self.get_states(self.environment.tools.db, self.environment.tools.db.time)
                    completeness_feedback = self.verifier.check_on_stop(self.trajectory, remaining - 1, new_orders=states["new_states"], old_orders=states["old_states"])
                    if completeness_feedback:
                        self._completeness_retries += 1
                    else:
                        self.done = True
                        self.termination_reason = TerminationReason.AGENT_STOP
                else:
                    self.done = True
                    self.termination_reason = TerminationReason.AGENT_STOP
            self.trajectory.append(agent_msg)
            self.message = agent_msg
            self.from_role = Role.AGENT
            if completeness_feedback:
                # Inject a matching fake tool_call into the agent message
                # so its valid tool_use → tool_result pair
                check_id = f"completeness_check_{self._completeness_retries}"
                if agent_msg.tool_calls is None:
                    agent_msg.tool_calls = []
                agent_msg.tool_calls.append(ToolCall(
                    id=check_id,
                    name="completeness_check",
                    arguments={},
                    requestor="assistant",
                ))
                feedback_tool_msg = ToolMessage(
                    id=check_id,
                    name="completeness_check",
                    content=completeness_feedback.strip(),
                    requestor="assistant",
                    role="tool",
                    error=True,
                )
                self.trajectory.append(feedback_tool_msg)
                self.message = feedback_tool_msg
                self.from_role = Role.ENV
                self.to_role = Role.AGENT
            elif agent_msg.is_tool_call():
                self.to_role = Role.ENV
            elif self.solo_mode and not self.agent.is_stop(agent_msg):
                # Agent tried to talk to a user that isn't there — nudge it back
                logger.warning(f"Solo mode agent produced non-tool-call text, nudging back: {agent_msg}")
                nudge_msg = UserMessage(
                    role="user",
                    content=(
                        "There is no user to communicate with. "
                        "You are operating solo — refer to the task instructions in previous messages. "
                        "Continue making tool calls to complete the task, or call ###STOP### if done."
                    ),
                )
                self.trajectory.append(nudge_msg)
                self.message = nudge_msg
                self.from_role = Role.USER
                self.to_role = Role.AGENT
            else:
                self.to_role = Role.USER
        elif self.from_role in [Role.AGENT, Role.USER] and self.to_role == Role.ENV:
            if not self.message.is_tool_call():
                raise ValueError("Agent or User should send tool call to environment")
            # Exclude the current agent message from the base trajectory so the judge
            # does not see all sibling tool calls as already-executed prior history.
            trajectory_before_current = self.trajectory[:-1]
            tool_msgs = []
            for tool_call in self.message.tool_calls:
                soundness_feedback = None
                if self.verifier:
                    # Build context = prior trajectory + a mock assistant message containing
                    # only the already-executed sibling tool_calls (so their IDs are registered
                    # by _extract_tool_history) + their real responses.
                    if tool_msgs:
                        mock_assistant = deepcopy(self.message)
                        mock_assistant.tool_calls = self.message.tool_calls[:len(tool_msgs)]
                        trajectory_for_judge = trajectory_before_current + [mock_assistant] + tool_msgs
                    else:
                        trajectory_for_judge = trajectory_before_current
                    soundness_feedback = self.verifier.check_soundness(
                        tool_call.name, tool_call.arguments, trajectory_for_judge
                    )
                    if soundness_feedback:  # Blocking behavior: return feedback as error, skip execution
                        tool_msgs.append(ToolMessage(
                            id=tool_call.id,
                            name=tool_call.name,
                            content=soundness_feedback,
                            requestor=tool_call.requestor,
                            role="tool",
                            error=True,
                        ))
                        continue
                tool_msg = self.environment.get_response(tool_call)
                # OLD: nudge behavior — prepend feedback to tool response
                # if soundness_feedback:
                #     tool_msg.content = (
                #         soundness_feedback.strip()
                #         + "\n\n"
                #         + (tool_msg.content or "")
                #     )
                tool_msgs.append(tool_msg)
                # Feed tool response to harness memory (if the verifier supports it)
                if self.verifier and hasattr(self.verifier, "observe_tool_response"):
                    self.verifier.observe_tool_response(
                        tool_call.name, tool_call.arguments or {}, tool_msg.content or ""
                    )
                # Increment error count if tool call failed
                if tool_msg.error:
                    self.num_errors += 1
            assert len(self.message.tool_calls) == len(tool_msgs), (
                "Number of tool calls and tool messages should be the same"
            )
            self.trajectory.extend(tool_msgs)
            if (
                len(tool_msgs) > 1
            ):
                self.message = MultiToolMessage(
                    role="tool",
                    tool_messages=tool_msgs,
                )
            else:
                self.message = tool_msgs[0]
            self.to_role = self.from_role
            self.from_role = Role.ENV
        else:
            raise ValueError(
                f"Invalid role combination. From role: {self.from_role}, To role: {self.to_role}"
            )
        self.step_count += 1

    def get_trajectory(self) -> list[Message]:
        """
        Get the trajectory of the simulation.
        The trajectory is sorted by timestamp, turn_idx are added to messages, trajectory is returned.
        """
        messages: list[Message] = sorted(
            deepcopy(self.trajectory),
            key=lambda x: x.timestamp,
        )
        trajectory = []
        for i, msg in enumerate(messages):
            msg = deepcopy(msg)
            msg.turn_idx = i
            trajectory.append(msg)
        return trajectory
    
    def get_states(self, db: DB, env_time: str) -> Dict[str, Any]:
        """
        Split the states into a dictionary.
        """
        from vita.utils import str_to_datetime

        states = []
        if hasattr(db, "orders"):
            states += list(db.orders.values())
        if hasattr(db, "books"):
            states += list(db.books.values())
        if hasattr(db, "reservations"):
            states += list(db.reservations.values())

        states_dict = {"old_states": [], "new_states": []}
        for state in states:
            if str_to_datetime(state.update_time) < str_to_datetime(env_time):
                states_dict["old_states"].append(state)
            else:
                states_dict["new_states"].append(state)
        return states_dict

    @classmethod
    def validate_message_history(cls, message_history: list[Message]):
        """
        Validate a message history.
            - Should only contain AssistantMessage, UserMessage, ToolMessage
            - All assistant/user messages should be either to user or tool call, not both.
            - If n tool calls are made by a participant, exactly n tool messages should follow with requestor matching the participant.
        """
        num_expected_tool_messages = 0
        requestor = None
        for msg in message_history:
            if isinstance(msg, AssistantMessage) or isinstance(msg, UserMessage):
                msg.validate()
                if msg.is_tool_call():
                    if num_expected_tool_messages > 0:
                        raise ValueError(
                            f"{num_expected_tool_messages} tool messages are missing. Got {msg.role} message."
                        )
                    num_expected_tool_messages = len(msg.tool_calls)
                    requestor = msg.role
                else:
                    num_expected_tool_messages = 0
                    requestor = None
            elif isinstance(msg, ToolMessage):
                if num_expected_tool_messages == 0 or requestor is None:
                    raise ValueError("No tool messages expected.")
                if requestor != msg.requestor:
                    raise ValueError(
                        f"Got tool message from {msg.requestor}, expected {requestor}."
                    )
                num_expected_tool_messages -= 1
            else:
                raise ValueError(f"Invalid message type: {type(msg)}")

    def _add_timestamps(
        self, message_history: list[Message]
    ) -> list[tuple[str, Message]]:
        """
        Add timestamps to the message history.
        This is used to sort the messages by timestamp.
        """
        time_offset = datetime.now() - timedelta(seconds=len(message_history))
        for i, msg in enumerate(message_history):
            msg.timestamp = format_time(time_offset + timedelta(seconds=i))
        return message_history
