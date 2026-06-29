"""VitaBench overlay file — modified from the original VitaBench repo
(https://github.com/meituan-longcat/vitabench), at src/vita/user/user_simulator.py.
Everything is verbatim from the original except for the following changes:

1. Added a ``solo_user_message`` argument to ``DummyUser.__init__``.
2. Added the ``DummyUser.build(...)`` classmethod that resolves the opening
   message per ``solo_user_mode`` ('live' vs 'file').
3. Added ``_fixed_message(...)`` and a short-circuit in ``generate_next_message``
   that returns the pregenerated message without an LLM call.
"""
from typing import Optional, Tuple

from loguru import logger

from vita.data_model.message import (
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    UserMessage,
)

from vita.environment.tool import Tool
from vita.user.base import (
    OUT_OF_SCOPE,
    STOP,
    TRANSFER,
    BaseUser,
    UserState,
    ValidUserInputMessage,
    is_valid_user_history_message,
)
from vita.utils.llm_utils import generate
from vita.prompts import get_prompts


def get_global_user_sim_guidelines(language: str = None) -> str:
    """
    Get the global user simulator guidelines.

    Returns:
        The global user simulator guidelines.
    """
    prompts = get_prompts(language)
    return prompts.user_system_prompt


class UserSimulator(BaseUser):
    """Stateless implementation of a user simulator."""

    def __init__(
        self,
        tools: Optional[list[Tool]] = None,
        instructions: Optional[str] = None,
        persona: Optional[str] = None,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        language: str = None,
    ):
        super().__init__(instructions=instructions, llm=llm, llm_args=llm_args)
        self.tools = tools
        self.persona = persona
        self.language = language

    @property
    def global_simulation_guidelines(self) -> str:
        """
        The simulation guidelines for the user simulator.
        """
        return get_global_user_sim_guidelines(self.language)

    @property
    def system_prompt(self) -> str:
        """
        The system prompt for the user simulator.
        """
        if self.instructions is None:
            logger.warning("No instructions provided for user simulator")
        system_prompt = self.global_simulation_guidelines.format(
            persona=self.persona,
            instructions=self.instructions,
        )
        return system_prompt

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> UserState:
        """
        Get the initial state of the user simulator.
        """
        if message_history is None:
            message_history = []
        assert all(is_valid_user_history_message(m) for m in message_history), (
            "Invalid user message history. User messages must be of type UserMessage, AssistantMessage, or ToolMessage to User."
        )

        user_state = UserState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )
        return user_state

    @classmethod
    def is_stop(cls, message: UserMessage) -> bool:
        """
        Check if the message is a stop message.
        """
        if message.is_tool_call():
            return False
        assert message.content is not None
        return (
            STOP in message.content
            or TRANSFER in message.content
            or OUT_OF_SCOPE in message.content
        )

    def generate_next_message(
        self, message: ValidUserInputMessage, state: UserState
    ) -> Tuple[UserMessage, UserState]:
        return self._generate_next_message(message, state)

    def _generate_next_message(
        self, message: ValidUserInputMessage, state: UserState
    ) -> Tuple[UserMessage, UserState]:
        """Get the response from the user simulator.

        Args:
            message: The assistant or tool message.
            state: The user simulator's state.

        Returns:
            A tuple containing the user message and the updated user state.
        """
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        messages = state.system_messages + state.flip_roles()

        assistant_message = generate(
            model=self.llm,
            messages=messages,
            tools=self.tools,
            **self.llm_args,
        )

        user_response = assistant_message.content
        logger.debug(f"Response: {user_response}")

        user_message = UserMessage(
            role="user",
            content=user_response,
            cost=assistant_message.cost,
            usage=assistant_message.usage,
            raw_data=assistant_message.raw_data,
        )

        if assistant_message.tool_calls is not None:
            user_message.tool_calls = []
            for tool_call in assistant_message.tool_calls:
                user_message.tool_calls.append(
                    ToolCall(
                        id=tool_call.id,
                        name=tool_call.name,
                        arguments=tool_call.arguments,
                        requestor="user",
                    )
                )

        state.messages.append(user_message)
        return user_message, state


class DummyUser(BaseUser):
    """A dummy user to run a agent solo simulation."""
    def __init__(
            self,
            tools: Optional[list[Tool]] = None,
            instructions: Optional[str] = None,
            persona: Optional[str] = None,
            llm: Optional[str] = None,
            llm_args: Optional[dict] = None,
            language: str = None,
            solo_user_message: Optional[str] = None,
    ):
        super().__init__(instructions=instructions, llm=llm, llm_args=llm_args)
        self.tools = tools
        self.persona = persona
        self.language = language
        self.solo_user_message = solo_user_message

    @classmethod
    def build(
            cls,
            task_id: str,
            instructions: str,
            persona: str,
            llm: Optional[str] = None,
            llm_args: Optional[dict] = None,
            language: str = None,
            solo_user_mode: str = "live",
            solo_user_messages: Optional[dict] = None,
    ) -> "DummyUser":
        """Construct a DummyUser, resolving the user message according to the given mode.

        Args:
            task_id: ID of the task being run, used to look up the pregenerated message.
            solo_user_mode: ``'live'`` — generate the opening message via LLM each run
                (introduces variance); ``'file'`` — look up a pregenerated message from
                *solo_user_messages* (deterministic, raises if the task is missing).
            solo_user_messages: Mapping of task_id -> pregenerated message string.
                Required when *solo_user_mode* is ``'file'``.
        """
        if solo_user_mode == "file":
            if solo_user_messages is None or task_id not in solo_user_messages:
                raise ValueError(
                    f"solo_user_mode='file' but no pregenerated message found for task '{task_id}'. "
                    "Run the pregeneration script first or switch to solo_user_mode='live'."
                )
            solo_user_message = solo_user_messages[task_id]
        else:
            solo_user_message = None

        return cls(
            instructions=instructions,
            persona=persona,
            llm=llm,
            llm_args=llm_args,
            language=language,
            solo_user_message=solo_user_message,
        )

    @property
    def system_prompt(self) -> str:
        prompts = get_prompts(self.language)
        return prompts.dummy_user_system_prompt.format(persona=self.persona, instructions=self.instructions)

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> UserState:
        """
        Get the initial state of the user simulator.
        """
        if message_history is None:
            message_history = []
        assert all(is_valid_user_history_message(m) for m in message_history), (
            "Invalid user message history. User messages must be of type UserMessage, AssistantMessage, or ToolMessage to User."
        )

        user_state = UserState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )
        return user_state

    @classmethod
    def is_stop(cls, message: UserMessage) -> bool:
        """
        Check if the message is a stop message.
        """
        if message.is_tool_call():
            return False
        assert message.content is not None
        return (
                STOP in message.content
                or TRANSFER in message.content
                or OUT_OF_SCOPE in message.content
        )

    def generate_next_message(
            self, message: ValidUserInputMessage, state: UserState
    ) -> Tuple[UserMessage, UserState]:
        if self.solo_user_message is not None:
            return self._fixed_message(message, state)
        return self._generate_next_message(message, state)

    def _fixed_message(
            self, message: ValidUserInputMessage, state: UserState
    ) -> Tuple[UserMessage, UserState]:
        """Return the pregenerated fixed message without calling the LLM."""
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        user_message = UserMessage(role="user", content=self.solo_user_message, cost=0.0)
        state.messages.append(user_message)
        return user_message, state

    def _generate_next_message(
            self, message: ValidUserInputMessage, state: UserState
    ) -> Tuple[UserMessage, UserState]:
        """Get the response from the user simulator.

        Args:
            message: The assistant or tool message.
            state: The user simulator's state.

        Returns:
            A tuple containing the user message and the updated user state.
        """
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        messages = state.system_messages + state.flip_roles()

        assistant_message = generate(
            model=self.llm,
            messages=messages,
            tools=self.tools,
            **self.llm_args,
        )

        user_response = assistant_message.content
        logger.debug(f"Response: {user_response}")

        user_message = UserMessage(
            role="user",
            content=user_response,
            cost=assistant_message.cost,
            usage=assistant_message.usage,
            raw_data=assistant_message.raw_data,
        )

        if assistant_message.tool_calls is not None:
            user_message.tool_calls = []
            for tool_call in assistant_message.tool_calls:
                user_message.tool_calls.append(
                    ToolCall(
                        id=tool_call.id,
                        name=tool_call.name,
                        arguments=tool_call.arguments,
                        requestor="user",
                    )
                )

        state.messages.append(user_message)
        return user_message, state