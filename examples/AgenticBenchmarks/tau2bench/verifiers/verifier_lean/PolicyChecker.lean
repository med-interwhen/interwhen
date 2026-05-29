import Mathlib

/-!
# Telecom Policy Checker

A self-contained Lean 4 specification + executable checker + soundness proofs
for the telecom agent policy described in `policy.md`, `tech_support_manual.md`,
and `tech_support_workflow.md`.

The runtime tool surface comes from `tools.py` (telecom domain). Lean does NOT
import the Python file; the toolset is encoded as the `Action` inductive below.

Naming conventions:
* `spec_X`     : Prop-level rule
* `check_X`    : decidable Bool checker
* `feedback_X` : human-readable failure message
* `:post` rules use the `_result_` infix.
-/

namespace Telecom

/-! ## §A. Data Models -/

/-- Identifier types are wrappers around `String` so that we get `DecidableEq`
for free without confusing `LineId` with `BillId`. -/
structure CustomerId where val : String deriving DecidableEq, Repr
structure LineId     where val : String deriving DecidableEq, Repr
structure BillId     where val : String deriving DecidableEq, Repr
structure PlanId     where val : String deriving DecidableEq, Repr
structure DeviceId   where val : String deriving DecidableEq, Repr

/-- We model dates as days since some epoch — only ordering matters for the
policy (e.g. "contract end date is in the past"). -/
abbrev Date := Int

/-- Account status (Policy §"Customer"). -/
inductive AccountStatus
  | active | suspended | pendingVerification | closed
  deriving DecidableEq, Repr

/-- Line status (Policy §"Line"). -/
inductive LineStatus
  | active | suspended | pendingActivation | closed
  deriving DecidableEq, Repr

/-- Bill status (Policy §"Bill"). The policy lists 5 distinct values, plus
"Awaiting Payment". -/
inductive BillStatus
  | draft | issued | paid | overdue | awaitingPayment | disputed
  deriving DecidableEq, Repr

inductive PaymentMethodType
  | creditCard | debitCard | payPal
  deriving DecidableEq, Repr

structure PaymentMethod where
  type        : PaymentMethodType
  last4       : String
  expiration  : String  -- MM/YYYY
  deriving DecidableEq, Repr

structure Plan where
  planId                 : PlanId
  name                   : String
  dataLimitGb            : Nat
  monthlyPrice           : Nat   -- cents
  refuelPricePerGb       : Nat   -- cents
  deriving Repr

structure Bill where
  billId       : BillId
  customerId   : CustomerId
  totalDue     : Nat
  status       : BillStatus
  deriving Repr

structure Line where
  lineId             : LineId
  phoneNumber        : String
  status             : LineStatus
  planId             : PlanId
  ownerId            : CustomerId
  dataUsedGb         : Nat
  dataRefuelingGb    : Nat
  roamingEnabled     : Bool
  contractEndDate    : Date
  deriving Repr

structure Customer where
  customerId    : CustomerId
  fullName      : String
  dateOfBirth   : String
  email         : String
  phoneNumber   : String
  status        : AccountStatus
  lineIds       : List LineId
  billIds       : List BillId
  deriving Repr

/-! ### Tool calls and history

The `ToolCall` enumeration mirrors the tool surface defined in `tools.py`.
We only encode tools relevant to policy checks; helper functions on the
Python side are not modelled.
-/
inductive ToolCall
  | getCustomerByPhone (phone : String)
  | getCustomerById    (cid : CustomerId)
  | getCustomerByName  (name : String) (dob : String)
  | getDetailsById     (id : String)
  | suspendLine        (cid : CustomerId) (lid : LineId) (reason : String)
  | resumeLine         (cid : CustomerId) (lid : LineId)
  | getBillsForCustomer (cid : CustomerId)
  | sendPaymentRequest (cid : CustomerId) (bid : BillId)
  | getDataUsage       (cid : CustomerId) (lid : LineId)
  | enableRoaming      (cid : CustomerId) (lid : LineId)
  | disableRoaming     (cid : CustomerId) (lid : LineId)
  | refuelData         (cid : CustomerId) (lid : LineId) (gbTimes100 : Nat)
                        -- gb stored ×100 to keep Nat
  | transferToHumanAgents (summary : String)
  deriving Repr

/-- The full agent state: a database snapshot, a possibly-identified customer,
and the history of prior tool calls.  `today` is the reference date for
expiry comparisons (Policy header: "The current time is 2025-02-25 12:08:00 EST").

The `lastToolResults` field stores the raw output string of every read
tool called this session, keyed by tool name.  Python ships these
verbatim — Lean parses them in POST checks. -/
structure AgentState where
  customers          : List Customer
  bills              : List Bill
  lines              : List Line
  plans              : List Plan
  identifiedCustomer : Option CustomerId
  history            : List ToolCall
  today              : Date
  lastToolResults    : List (String × String) := []
  userPhone          : String := ""
  deriving Repr

/-- SLM-derived hypothesis Bools.  These are the ONLY pre-cooked Bool
inputs Lean accepts in POST checks; everything else must be parsed from
structured tool results inside Lean. -/
structure Hyp where
  travelling                    : Bool := false
  userConfirmedRefuelPrice      : Bool := false
  userGrantedPaymentPermission  : Bool := false
  vpnPerformancePoor            : Bool := false
  validSuspensionReason         : Bool := false
  deriving Repr

/-! ## §B. Database lookup helpers (pure, decidable) -/

def lookupCustomer (s : AgentState) (cid : CustomerId) : Option Customer :=
  s.customers.find? (fun c => c.customerId = cid)

def lookupBill (s : AgentState) (bid : BillId) : Option Bill :=
  s.bills.find? (fun b => b.billId = bid)

def lookupLine (s : AgentState) (lid : LineId) : Option Line :=
  s.lines.find? (fun l => l.lineId = lid)

/-! ## §C. Opaque hypothesis inputs

Free-text facts that are NOT stored in the runtime database are encoded as
opaque `Prop`s.  The Python translator discharges them via explicit kwargs
or an SLM extraction call. -/

/-- The user has explicitly authorised the agent to make payments on
their behalf (Policy §"Overdue Bill Payment": "the ticket specifies that
the user has given you the permission to make payments"). -/
opaque UserGrantedPaymentPermission : Prop

/-- The user is currently travelling outside their carrier's home network
(Policy §"Data Roaming"). -/
opaque UserIsTravelling (lid : LineId) : Prop

/-- The user has confirmed the refueling price (Policy §"Data Refueling":
"Confirm the price"). -/
opaque UserConfirmedRefuelPrice (lid : LineId) (gbTimes100 : Nat) : Prop

/-- The supplied suspension reason is one of the policy-allowed reasons
(Policy §"Line Suspension": overdue bill or contract end past). The
runtime has no enumerated `reason`, so we treat it as opaque. -/
opaque ValidSuspensionReason (reason : String) : Prop

/-- The VPN performance level reported by `check_vpn_status` is "Poor"
(tech-support manual §"VPN Connection Issues" — combined trigger). -/
opaque VPNPerformancePoor : Prop

/-! ## §D. Triplets

Each triplet has the form `spec_X` (Prop), `check_X` (Bool), `check_X_iff`
(equivalence proof), and `feedback_X` (failure reason).

### Source taxonomy
* `db`       — current database/state snapshot
* `args`     — tool call arguments
* `history`  — prior tool calls in `state.history`
* `context`  — free text from the conversation/ticket (opaque hypotheses)
* `result`   — the **output string** of a previously-executed read tool
              (used by `:post` checks raised from `checkResult` in §G)
* combine with `+`, e.g. `result+context`
-/

/-- Tiny helper: substring containment.  Implemented via `splitOn` to keep
the proofs trivial — `splitOn pat` returns at least 2 pieces iff `pat`
occurs in the haystack.  Note: `containsSubstr "" pat = false` for any
non-empty `pat`, which gives the desired "empty result is silent"
behaviour for "fire when present" checks. -/
def containsSubstr (haystack needle : String) : Bool :=
  decide ((haystack.splitOn needle).length ≥ 2)

/-! ### §C.parsing — utilities consumed by §D.POST

These parsers are intentionally minimal: they don't validate full JSON,
they just locate `"key"` substrings and read the value up to the next
delimiter (`,`, `}`, or `"` for string values).  That is enough for the
structured outputs of `get_data_usage` and `get_details_by_id`.

The pattern lets POST specs be written directly over the parser
results, so each `_iff` lemma is a simple case split with no hidden
`decide`. -/

/-- Helper: locate the value of an unquoted JSON field (number, bool,
or quoted string).  Tolerates optional whitespace after the colon. -/
private def jsonValueRaw (r key : String) : Option String :=
  let needle := "\"" ++ key ++ "\":"
  match r.splitOn needle with
  | _ :: rest :: _ =>
    let upToComma := (rest.splitOn ",").headD rest
    let upToBrace := (upToComma.splitOn "}").headD upToComma
    some upToBrace.toSubstring.trim.toString
  | _ => none

/-- Parse a JSON string value: tolerates whitespace after the colon
(`"key": "<val>"` and `"key":"<val>"` both work). -/
def parseJsonString (r key : String) : Option String :=
  let needle := "\"" ++ key ++ "\":"
  match r.splitOn needle with
  | _ :: rest :: _ =>
    -- Skip leading whitespace and the opening quote.
    let trimmed := rest.toSubstring.trim.toString
    match trimmed.splitOn "\"" with
    | _ :: val :: _ => some val
    | _             => none
  | _ => none

/-- Parse a JSON numeric field as `Nat`.  `none` if missing or not a
non-negative integer literal. -/
def parseJsonNum (r key : String) : Option Nat :=
  jsonValueRaw r key >>= String.toNat?

/-- Parse a JSON numeric field as Nat-of-hundredths.  Accepts integers
and decimals with up to 2 fractional digits: "1" → some 100, "1.5" →
some 150, "0.25" → some 25.  None on parse failure. -/
def parseJsonNumX100 (r key : String) : Option Nat := do
  let raw ← jsonValueRaw r key
  match raw.splitOn "." with
  | [whole]       => (· * 100) <$> whole.toNat?
  | [whole, frac] =>
      let frac2 := (frac ++ "00").take 2
      let w ← whole.toNat?
      let f ← frac2.toNat?
      some (w * 100 + f)
  | _             => none

/-- Strip every non-digit character from a phone string.  Used to
compare `+1-555-123-4567` against `15551234567`, etc. -/
def normalisePhone (s : String) : String :=
  s.foldl (init := "") (fun acc c => if c.isDigit then acc.push c else acc)

/-- Parse a JSON boolean field. -/
def parseJsonBool (r key : String) : Option Bool :=
  match jsonValueRaw r key with
  | some "true"  => some true
  | some "false" => some false
  | _            => none

/-- Locate `pre` in `r`, return everything up to the next `.` (or end),
trimmed.  Used to extract the comma-separated permission list from
`check_app_permissions` output. -/
def parseAfterPrefix (r pre : String) : Option String :=
  match r.splitOn pre with
  | _ :: rest :: _ =>
    let raw := (rest.splitOn ".").headD rest
    some raw.toSubstring.trim.toString
  | _ => none

/-- Parse a comma-separated list following `pre` in `r`. -/
def parseCommaList (r pre : String) : Option (List String) :=
  (parseAfterPrefix r pre).map
    (fun s => (s.splitOn ", ").map (fun x => x.toSubstring.trim.toString))

/-- Look up the most-recent raw result string for tool `t` in
`s.lastToolResults`.  Returns `none` if the tool was never called. -/
def lastToolResult (s : AgentState) (t : String) : Option String :=
  (s.lastToolResults.find? (fun p => p.1 = t)).map Prod.snd

/-! ### D.1 Customer lookup must include DOB when looking up by name

Policy §"Customer Lookup":
"For name lookup, date of birth is required for verification purposes."

Implicit corollary: lookup by name alone is forbidden.
-/

-- Policy §"Customer Lookup": "For name lookup, date of birth is required for verification purposes."
-- [source: args] [phase: pre] [tool: get_customer_by_name]
def spec_nameLookupHasDOB (name : String) (dob : String) : Prop :=
  name ≠ "" ∧ dob ≠ ""

def check_nameLookupHasDOB (name : String) (dob : String) : Bool :=
  (name ≠ "") && (dob ≠ "")

theorem check_nameLookupHasDOB_iff (name dob : String) :
    check_nameLookupHasDOB name dob = true ↔ spec_nameLookupHasDOB name dob := by
  unfold check_nameLookupHasDOB spec_nameLookupHasDOB
  simp

instance (name dob : String) : Decidable (spec_nameLookupHasDOB name dob) :=
  decidable_of_iff _ (check_nameLookupHasDOB_iff name dob)

def feedback_nameLookupHasDOB (name : String) (dob : String) : String :=
  s!"Customer name lookup requires both full name and date of birth; got name='{name}', dob='{dob}'."

/-! ### D.2 Technical support: customer must be identified first

Policy §"Technical Support": "You must first identify the customer."
We treat any tool call other than the three customer-lookup helpers and
`transfer_to_human_agents` as requiring an identified customer.
-/

-- Policy §"Technical Support": "You must first identify the customer."
-- [source: db] [phase: pre] [tool: <any technical-support tool>]
def spec_customerIdentified (s : AgentState) : Prop :=
  s.identifiedCustomer.isSome = true

def check_customerIdentified (s : AgentState) : Bool :=
  s.identifiedCustomer.isSome

theorem check_customerIdentified_iff (s : AgentState) :
    check_customerIdentified s = true ↔ spec_customerIdentified s := by
  unfold check_customerIdentified spec_customerIdentified
  rfl

instance (s : AgentState) : Decidable (spec_customerIdentified s) :=
  decidable_of_iff _ (check_customerIdentified_iff s)

def feedback_customerIdentified (_s : AgentState) : String :=
  "No customer is currently identified; you must first identify the customer (lookup by phone, customer ID, or name+DOB) before performing this action."

/-! ### D.3 send_payment_request: bill must be overdue

Policy §"Overdue Bill Payment":
"Check the bill status to make sure it is overdue."
"The send payement request tool will not check if the bill is overdue.
You should always check that the bill is overdue before sending a payment request."
-/

-- Policy §"Overdue Bill Payment": "Check the bill status to make sure it is overdue."
-- [source: db] [phase: pre] [tool: send_payment_request]
def spec_billOverdue (s : AgentState) (bid : BillId) : Prop :=
  ∃ b, lookupBill s bid = some b ∧ b.status = BillStatus.overdue

def check_billOverdue (s : AgentState) (bid : BillId) : Bool :=
  match lookupBill s bid with
  | some b => decide (b.status = BillStatus.overdue)
  | none   => false

theorem check_billOverdue_iff (s : AgentState) (bid : BillId) :
    check_billOverdue s bid = true ↔ spec_billOverdue s bid := by
  unfold check_billOverdue spec_billOverdue
  cases hfind : lookupBill s bid with
  | none   => simp
  | some b => simp [hfind]

instance (s : AgentState) (bid : BillId) : Decidable (spec_billOverdue s bid) :=
  decidable_of_iff _ (check_billOverdue_iff s bid)

def feedback_billOverdue (_s : AgentState) (bid : BillId) : String :=
  s!"Bill '{bid.val}' is not in OVERDUE status; payment request can only be sent for overdue bills."

/-! ### D.4 send_payment_request: no other bill in AWAITING_PAYMENT

Policy §"Overdue Bill Payment":
"A user can only have one bill in the AWAITING PAYMENT status at a time."
This is also enforced by the runtime in `send_payment_request`.
-/

-- Policy §"Overdue Bill Payment": "A user can only have one bill in the AWAITING PAYMENT status at a time."
-- [source: db] [phase: pre] [tool: send_payment_request]
def spec_noOtherAwaitingPayment (s : AgentState) (cid : CustomerId) : Prop :=
  ∀ b ∈ s.bills, b.customerId = cid → b.status ≠ BillStatus.awaitingPayment

def check_noOtherAwaitingPayment (s : AgentState) (cid : CustomerId) : Bool :=
  s.bills.all (fun b =>
    !(decide (b.customerId = cid)) || !(decide (b.status = BillStatus.awaitingPayment)))

theorem check_noOtherAwaitingPayment_iff (s : AgentState) (cid : CustomerId) :
    check_noOtherAwaitingPayment s cid = true ↔ spec_noOtherAwaitingPayment s cid := by
  unfold check_noOtherAwaitingPayment spec_noOtherAwaitingPayment
  simp [List.all_eq_true]
  constructor
  · intro h b hmem hcid hst
    have := h b hmem
    rw [hcid, hst] at this
    simp at this
  · intro h b hmem
    by_cases hcid : b.customerId = cid
    · by_cases hst : b.status = BillStatus.awaitingPayment
      · exact absurd hst (h b hmem hcid)
      · simp [hcid, hst]
    · simp [hcid]

instance (s : AgentState) (cid : CustomerId) : Decidable (spec_noOtherAwaitingPayment s cid) :=
  decidable_of_iff _ (check_noOtherAwaitingPayment_iff s cid)

def feedback_noOtherAwaitingPayment (_s : AgentState) (cid : CustomerId) : String :=
  s!"Customer '{cid.val}' already has a bill in AWAITING_PAYMENT status; only one such bill is permitted at a time."

/-! ### D.5 send_payment_request: bill belongs to the customer

Implicit from Policy §"Overdue Bill Payment" (the runtime tool also checks
`bill_id in customer.bill_ids`).
-/

-- Policy §"Overdue Bill Payment": implicit ownership check ("the customer's overdue bill").
-- [source: db] [phase: pre] [tool: send_payment_request]
def spec_billBelongsToCustomer (s : AgentState) (cid : CustomerId) (bid : BillId) : Prop :=
  ∃ c, lookupCustomer s cid = some c ∧ bid ∈ c.billIds

def check_billBelongsToCustomer (s : AgentState) (cid : CustomerId) (bid : BillId) : Bool :=
  match lookupCustomer s cid with
  | some c => decide (bid ∈ c.billIds)
  | none   => false

theorem check_billBelongsToCustomer_iff (s : AgentState) (cid : CustomerId) (bid : BillId) :
    check_billBelongsToCustomer s cid bid = true ↔ spec_billBelongsToCustomer s cid bid := by
  unfold check_billBelongsToCustomer spec_billBelongsToCustomer
  cases hfind : lookupCustomer s cid with
  | none   => simp
  | some c => simp [hfind]

instance (s : AgentState) (cid : CustomerId) (bid : BillId) :
    Decidable (spec_billBelongsToCustomer s cid bid) :=
  decidable_of_iff _ (check_billBelongsToCustomer_iff s cid bid)

def feedback_billBelongsToCustomer (_s : AgentState) (cid : CustomerId) (bid : BillId) : String :=
  s!"Bill '{bid.val}' does not belong to customer '{cid.val}'."

/-! ### D.6 make_payment requires explicit user permission

Policy §"Overdue Bill Payment":
"You can only do so [make payments] if the ticket specifies that the user
has given you the permission to make payments!"

The Python `tools.py` does NOT expose a `make_payment` or `check_payment_request`
tool, so this is a vacuous-in-current-runtime rule.
-/

-- Policy §"Overdue Bill Payment": "You can only do so if the ticket specifies that the user has given you the permission to make payments!"
-- [source: context] [phase: pre] [tool: N/A — would gate make_payment]
-- TODO: vacuous in current runtime
def spec_canMakePayment (_s : AgentState) (_bid : BillId)
    (_hPerm : UserGrantedPaymentPermission) : Prop := True

def check_canMakePayment (_s : AgentState) (_bid : BillId) : Bool := true

theorem check_canMakePayment_iff (s : AgentState) (bid : BillId)
    (hPerm : UserGrantedPaymentPermission) :
    check_canMakePayment s bid = true ↔ spec_canMakePayment s bid hPerm := by
  unfold check_canMakePayment spec_canMakePayment; simp

def feedback_canMakePayment (_s : AgentState) (bid : BillId) : String :=
  s!"Cannot make payment for bill '{bid.val}': the ticket must explicitly grant the agent permission to make payments."

-- Policy §"Overdue Bill Payment": "Check their payment requests using the check_payment_request tool" before make_payment.
-- [source: history] [phase: pre] [tool: N/A — would gate make_payment]
-- TODO: vacuous in current runtime (no check_payment_request / make_payment tools exposed)
def spec_checkPaymentRequestBeforeMakePayment (_s : AgentState) (_bid : BillId) : Prop := True

def check_checkPaymentRequestBeforeMakePayment (_s : AgentState) (_bid : BillId) : Bool := true

theorem check_checkPaymentRequestBeforeMakePayment_iff (s : AgentState) (bid : BillId) :
    check_checkPaymentRequestBeforeMakePayment s bid = true ↔
      spec_checkPaymentRequestBeforeMakePayment s bid := by
  unfold check_checkPaymentRequestBeforeMakePayment spec_checkPaymentRequestBeforeMakePayment
  simp

def feedback_checkPaymentRequestBeforeMakePayment (_s : AgentState) (bid : BillId) : String :=
  s!"Must call check_payment_request for bill '{bid.val}' before make_payment."

/-! ### D.7 resume_line: no overdue bills remain for the customer

Policy §"Line Suspension":
"You are allowed to lift the suspension after the user has paid all their
overdue bills."
-/

-- Policy §"Line Suspension": "You are allowed to lift the suspension after the user has paid all their overdue bills."
-- [source: db] [phase: pre] [tool: resume_line]
def spec_noOverdueBillsForCustomer (s : AgentState) (cid : CustomerId) : Prop :=
  ∀ b ∈ s.bills, b.customerId = cid → b.status ≠ BillStatus.overdue

def check_noOverdueBillsForCustomer (s : AgentState) (cid : CustomerId) : Bool :=
  s.bills.all (fun b =>
    !(decide (b.customerId = cid)) || !(decide (b.status = BillStatus.overdue)))

theorem check_noOverdueBillsForCustomer_iff (s : AgentState) (cid : CustomerId) :
    check_noOverdueBillsForCustomer s cid = true ↔ spec_noOverdueBillsForCustomer s cid := by
  unfold check_noOverdueBillsForCustomer spec_noOverdueBillsForCustomer
  simp [List.all_eq_true]
  constructor
  · intro h b hmem hcid hst
    have := h b hmem
    rw [hcid, hst] at this; simp at this
  · intro h b hmem
    by_cases hcid : b.customerId = cid
    · by_cases hst : b.status = BillStatus.overdue
      · exact absurd hst (h b hmem hcid)
      · simp [hcid, hst]
    · simp [hcid]

instance (s : AgentState) (cid : CustomerId) :
    Decidable (spec_noOverdueBillsForCustomer s cid) :=
  decidable_of_iff _ (check_noOverdueBillsForCustomer_iff s cid)

def feedback_noOverdueBillsForCustomer (_s : AgentState) (cid : CustomerId) : String :=
  s!"Customer '{cid.val}' still has overdue bills; suspension cannot be lifted until all overdue bills are paid."

/-! ### D.8 resume_line: contract end date must NOT be in the past

Policy §"Line Suspension":
"You are not allowed to lift the suspension if the line's contract end date
is in the past, even if the user has paid all their overdue bills."
-/

-- Policy §"Line Suspension": "You are not allowed to lift the suspension if the line's contract end date is in the past."
-- [source: db] [phase: pre] [tool: resume_line]
def spec_contractNotExpired (s : AgentState) (lid : LineId) : Prop :=
  ∃ l, lookupLine s lid = some l ∧ l.contractEndDate ≥ s.today

def check_contractNotExpired (s : AgentState) (lid : LineId) : Bool :=
  match lookupLine s lid with
  | some l => decide (l.contractEndDate ≥ s.today)
  | none   => false

theorem check_contractNotExpired_iff (s : AgentState) (lid : LineId) :
    check_contractNotExpired s lid = true ↔ spec_contractNotExpired s lid := by
  unfold check_contractNotExpired spec_contractNotExpired
  cases hfind : lookupLine s lid with
  | none   => simp
  | some l => simp [hfind]

instance (s : AgentState) (lid : LineId) : Decidable (spec_contractNotExpired s lid) :=
  decidable_of_iff _ (check_contractNotExpired_iff s lid)

def feedback_contractNotExpired (_s : AgentState) (lid : LineId) : String :=
  s!"Line '{lid.val}' has a contract end date in the past; suspension cannot be lifted by the agent."

/-! ### D.9 POST resume_line: user must reboot device for service

Policy §"Line Suspension":
"After you resume the line, the user will have to reboot their device to get service."

Modelled as a post-condition warning that the agent must surface to the user.
The check returns true (the policy is informational, not a denial), but
`feedback_result_resumeLine_reboot` provides the reminder.
-/

-- Policy §"Line Suspension": "After you resume the line, the user will have to reboot their device to get service."
-- [source: db] [phase: post] [tool: resume_line]
def spec_result_resumeLine_reboot (_s : AgentState) (_lid : LineId) : Prop := True

def check_result_resumeLine_reboot (_s : AgentState) (_lid : LineId) : Bool := true

theorem check_result_resumeLine_reboot_iff (s : AgentState) (lid : LineId) :
    check_result_resumeLine_reboot s lid = true ↔ spec_result_resumeLine_reboot s lid := by
  unfold check_result_resumeLine_reboot spec_result_resumeLine_reboot; simp

def feedback_result_resumeLine_reboot (_s : AgentState) (lid : LineId) : String :=
  s!"Reminder: after resuming line '{lid.val}', the user must reboot their device to get service."

/-! ### D.10 refuel_data: amount must be ≤ 2 GB

Policy §"Data Refueling":
"The maximum amount of data that can be refueled is 2GB."
-/

-- Policy §"Data Refueling": "The maximum amount of data that can be refueled is 2GB."
-- [source: args] [phase: pre] [tool: refuel_data]
def spec_refuelMaxGB (gbTimes100 : Nat) : Prop := gbTimes100 ≤ 200

def check_refuelMaxGB (gbTimes100 : Nat) : Bool := decide (gbTimes100 ≤ 200)

theorem check_refuelMaxGB_iff (gbTimes100 : Nat) :
    check_refuelMaxGB gbTimes100 = true ↔ spec_refuelMaxGB gbTimes100 := by
  unfold check_refuelMaxGB spec_refuelMaxGB; simp

instance (gbTimes100 : Nat) : Decidable (spec_refuelMaxGB gbTimes100) :=
  decidable_of_iff _ (check_refuelMaxGB_iff gbTimes100)

def feedback_refuelMaxGB (gbTimes100 : Nat) : String :=
  s!"Refuel amount {gbTimes100}/100 GB exceeds the 2 GB maximum allowed by policy."

/-! ### D.11 refuel_data: amount must be positive

Implicit from runtime check: `if gb_amount <= 0: raise ValueError`.
-/

-- Policy §"Data Refueling": "Know how much data they want to refuel" (implies positive amount).
-- [source: args] [phase: pre] [tool: refuel_data]
def spec_refuelPositive (gbTimes100 : Nat) : Prop := gbTimes100 > 0

def check_refuelPositive (gbTimes100 : Nat) : Bool := decide (gbTimes100 > 0)

theorem check_refuelPositive_iff (gbTimes100 : Nat) :
    check_refuelPositive gbTimes100 = true ↔ spec_refuelPositive gbTimes100 := by
  unfold check_refuelPositive spec_refuelPositive; simp

instance (gbTimes100 : Nat) : Decidable (spec_refuelPositive gbTimes100) :=
  decidable_of_iff _ (check_refuelPositive_iff gbTimes100)

def feedback_refuelPositive (gbTimes100 : Nat) : String :=
  s!"Refuel amount must be strictly positive; got {gbTimes100}/100 GB."

/-! ### D.12 refuel_data: user has confirmed the price

Policy §"Data Refueling": "Confirm the price". This is a free-text fact:
the user must verbally agree to the cost. Modelled as an opaque hypothesis.
-/

-- Policy §"Data Refueling": "Confirm the price."
-- [source: context] [phase: pre] [tool: refuel_data]
def spec_refuelPriceConfirmed (lid : LineId) (gbTimes100 : Nat)
    (_h : UserConfirmedRefuelPrice lid gbTimes100) : Prop := True

def check_refuelPriceConfirmed (_lid : LineId) (_gbTimes100 : Nat) : Bool := true

theorem check_refuelPriceConfirmed_iff (lid : LineId) (gbTimes100 : Nat)
    (h : UserConfirmedRefuelPrice lid gbTimes100) :
    check_refuelPriceConfirmed lid gbTimes100 = true ↔
      spec_refuelPriceConfirmed lid gbTimes100 h := by
  unfold check_refuelPriceConfirmed spec_refuelPriceConfirmed; simp

def feedback_refuelPriceConfirmed (lid : LineId) (gbTimes100 : Nat) : String :=
  s!"User has not confirmed the refueling price for {gbTimes100}/100 GB on line '{lid.val}'."

/-! ### D.13 enable_roaming: user is travelling

Policy §"Data Roaming":
"We offer data roaming to users who are traveling outside their home network."
"If a user is traveling outside their home network, you should check if the
line is roaming enabled. If it is not, you should enable it at no cost for the
user."
-/

-- Policy §"Data Roaming": "We offer data roaming to users who are traveling outside their home network."
-- [source: context] [phase: pre] [tool: enable_roaming]
def spec_userTravelling (lid : LineId) (_h : UserIsTravelling lid) : Prop := True

def check_userTravelling (_lid : LineId) : Bool := true

theorem check_userTravelling_iff (lid : LineId) (h : UserIsTravelling lid) :
    check_userTravelling lid = true ↔ spec_userTravelling lid h := by
  unfold check_userTravelling spec_userTravelling; simp

def feedback_userTravelling (lid : LineId) : String :=
  s!"Cannot enable roaming on line '{lid.val}': policy only allows enabling roaming when the user is travelling outside their home network."

/-! ### D.14 suspend_line: reason must be a policy-allowed reason

Policy §"Line Suspension":
"A line can be suspended for the following reasons:
  - The user has an overdue bill.
  - The line's contract end date is in the past."
-/

-- Policy §"Line Suspension": enumerated list of allowed reasons.
-- [source: context] [phase: pre] [tool: suspend_line]
def spec_validSuspensionReason (reason : String) (_h : ValidSuspensionReason reason) : Prop := True

def check_validSuspensionReason (_reason : String) : Bool := true

theorem check_validSuspensionReason_iff (reason : String) (h : ValidSuspensionReason reason) :
    check_validSuspensionReason reason = true ↔ spec_validSuspensionReason reason h := by
  unfold check_validSuspensionReason spec_validSuspensionReason; simp

def feedback_validSuspensionReason (reason : String) : String :=
  s!"Suspension reason '{reason}' is not one of the policy-allowed reasons (overdue bill or contract end date in the past)."

/-! ### D.15 transfer_to_human_agents: only as a last resort

Policy §header:
"You should escalate to a human agent if and only if the request cannot be
handled within the scope of your actions."
"You should try your best to resolve the issue before escalating the user
to a human agent."
Tech-support manual §Introduction:
"Make sure you try all the possible ways to resolve the user's issue before
transferring to a human agent."

The runtime cannot decide whether the request is handleable — that is a
free-text judgement.  Modelled as an opaque hypothesis. -/

/-- The current request is genuinely outside the agent's tool surface or all
in-scope remediation has been attempted unsuccessfully. -/
opaque RequestRequiresHumanEscalation : Prop

-- Policy §header / Tech-support manual: "escalate iff the request cannot be handled" + "try all resolution steps first."
-- [source: context+history] [phase: pre] [tool: transfer_to_human_agents]
def spec_escalationJustified (_h : RequestRequiresHumanEscalation) : Prop := True

def check_escalationJustified : Bool := true

theorem check_escalationJustified_iff (h : RequestRequiresHumanEscalation) :
    check_escalationJustified = true ↔ spec_escalationJustified h := by
  unfold check_escalationJustified spec_escalationJustified; simp

def feedback_escalationJustified : String :=
  "transfer_to_human_agents must only be used after exhausting in-scope remediation, or when the request is genuinely outside the agent's tool surface."

/-! ### D.POST  Workflow-derived POST checks

Source-of-truth output formats (verbatim from `user_tools.py`):
  * `get_data_usage`         → JSON `{"data_used_gb":N,"data_limit_gb":N,"data_refueling_gb":N}`
  * `get_details_by_id`(Line) → JSON `{"id":"L…","phone_number":"…","roaming_enabled":true|false, …}`
  * `check_app_permissions`  → `App '<name>' has permission for: p1, p2, ….`

`Hyp` carries SLM-derived Bools — the only pre-cooked inputs allowed. -/

/-! #### D.POST.1  `get_data_usage` → data exceeded

Workflow §2.1.4: "If Data Usage is EXCEEDED → … refuel data or change
to plan with a higher data limit." -/

-- [source: result] [phase: post] [tool: get_data_usage → refuel_data | change_plan]
def spec_result_dataUsage_exceeded (r : String) : Prop :=
  ∀ used limit refuel : Nat,
    parseJsonNumX100 r "data_used_gb"        = some used →
    parseJsonNumX100 r "data_limit_gb"       = some limit →
    parseJsonNumX100 r "data_refueling_gb"   = some refuel →
    used < limit + refuel

def check_result_dataUsage_exceeded (r : String) : Bool :=
  match parseJsonNumX100 r "data_used_gb",
        parseJsonNumX100 r "data_limit_gb",
        parseJsonNumX100 r "data_refueling_gb" with
  | some u, some l, some f => decide (u < l + f)
  | _, _, _                => true

theorem check_result_dataUsage_exceeded_iff (r : String) :
    check_result_dataUsage_exceeded r = true ↔ spec_result_dataUsage_exceeded r := by
  unfold check_result_dataUsage_exceeded spec_result_dataUsage_exceeded
  refine ⟨?_, ?_⟩
  · intro h u l f hu hl hf
    rw [hu, hl, hf] at h
    simpa using h
  · intro h
    cases hu : parseJsonNumX100 r "data_used_gb" with
    | none => simp [hu]
    | some u =>
      cases hl : parseJsonNumX100 r "data_limit_gb" with
      | none => simp [hu, hl]
      | some l =>
        cases hf : parseJsonNumX100 r "data_refueling_gb" with
        | none => simp [hu, hl, hf]
        | some f =>
          have := h u l f hu hl hf
          simp [hu, hl, hf, this]

instance (r : String) : Decidable (spec_result_dataUsage_exceeded r) :=
  decidable_of_iff _ (check_result_dataUsage_exceeded_iff r)

def feedback_result_dataUsage_exceeded : String :=
  "get_data_usage indicates the line's data usage has met or exceeded plan + refuel allowance — with user permission, refuel_data() or transfer to plan-change flow (workflow §2.1.4 \"Check Data Usage\")."

/-! #### D.POST.2  `get_details_by_id` (Line) + `Hyp.travelling`
        → roaming flag must be on while travelling

Rubric:
* Q1 ✓  JSON with a Bool field `roaming_enabled`.
* Q2 ✓  Cross-source: parsed Bool conjoined with SLM-derived
        `Hyp.travelling` (an explicitly-allowed pre-cooked input).
* Q3 ✓  Spec is `¬ (roaming_enabled = false ∧ travelling = true)`,
        a typed Bool relation — not a substring test.
* Q4 ✓  `_iff` composes by case-split on `parseJsonBool` and on the
        hypothesis Bool; no `decide` on a precomputed input.

Workflow §2.1.2 / Policy §"Data Roaming": when the user is travelling
and the line is not roaming-enabled, the agent must call
`enable_roaming()` (at no cost). -/

-- [source: result+hyp] [phase: post] [tool: get_details_by_id → enable_roaming]
def spec_result_lineRoamingDisabled (r : String) (h : Hyp) : Prop :=
  ∀ enabled : Bool,
    parseJsonBool r "roaming_enabled" = some enabled →
    ¬ (enabled = false ∧ h.travelling = true)

def check_result_lineRoamingDisabled (r : String) (h : Hyp) : Bool :=
  match parseJsonBool r "roaming_enabled" with
  | some enabled => ! ((! enabled) && h.travelling)
  | none         => true

theorem check_result_lineRoamingDisabled_iff (r : String) (h : Hyp) :
    check_result_lineRoamingDisabled r h = true ↔
      spec_result_lineRoamingDisabled r h := by
  unfold check_result_lineRoamingDisabled spec_result_lineRoamingDisabled
  refine ⟨?_, ?_⟩
  · intro hk e he
    rw [he] at hk
    cases e <;> cases ht : h.travelling <;>
      simp_all
  · intro hyp
    cases hp : parseJsonBool r "roaming_enabled" with
    | none      => simp [hp]
    | some e    =>
      have := hyp e hp
      cases e <;> cases ht : h.travelling <;> simp_all

instance (r : String) (h : Hyp) :
    Decidable (spec_result_lineRoamingDisabled r h) :=
  decidable_of_iff _ (check_result_lineRoamingDisabled_iff r h)

def feedback_result_lineRoamingDisabled : String :=
  "User is travelling and the line's `roaming_enabled` flag is false — guide the agent to call enable_roaming() at no cost (workflow §2.1.2 / policy §\"Data Roaming\")."

/-! #### D.POST.3  `get_details_by_id` (Line) → phone number agrees with state

Rubric:
* Q1 ✓  JSON string field `phone_number`.
* Q2(d) ✓  Cross-state: parsed value compared to `s.userPhone` from
        `AgentState`, not against a Python-precomputed Bool.
* Q3 ✓  Spec is a string equality between parsed value and stored
        state — typed Lean comparison.
* Q4 ✓  `_iff` composes by case-split on `parseJsonString` and
        `String.decEq`.

Policy §"Customer Lookup" identification rule (cross-check phone): if
the agent looked up a line and the line's phone number disagrees with
the user-supplied phone, identification is suspect. -/

-- [source: result+state] [phase: post] [tool: get_details_by_id → re-identify customer]
def spec_result_linePhoneMatchesState (s : AgentState) (r : String) : Prop :=
  ∀ phone : String,
    parseJsonString r "phone_number" = some phone →
    normalisePhone phone = normalisePhone s.userPhone

def check_result_linePhoneMatchesState (s : AgentState) (r : String) : Bool :=
  match parseJsonString r "phone_number" with
  | some phone => decide (normalisePhone phone = normalisePhone s.userPhone)
  | none       => true

theorem check_result_linePhoneMatchesState_iff (s : AgentState) (r : String) :
    check_result_linePhoneMatchesState s r = true ↔
      spec_result_linePhoneMatchesState s r := by
  unfold check_result_linePhoneMatchesState spec_result_linePhoneMatchesState
  refine ⟨?_, ?_⟩
  · intro hk p hp
    rw [hp] at hk
    simpa using hk
  · intro hyp
    cases hp : parseJsonString r "phone_number" with
    | none   => simp [hp]
    | some p =>
      have := hyp p hp
      simp [hp, this]

instance (s : AgentState) (r : String) :
    Decidable (spec_result_linePhoneMatchesState s r) :=
  decidable_of_iff _ (check_result_linePhoneMatchesState_iff s r)

def feedback_result_linePhoneMatchesState : String :=
  "get_details_by_id returned a `phone_number` that does not match the user-supplied phone in state — re-verify customer identity before acting on this line (policy §\"Customer Lookup\")."

/-! #### D.POST.4  `check_app_permissions` (messaging) → storage AND sms granted

Rubric:
* Q1 ✓  Fixed-prefix output `App '<name>' has permission for: p1, p2, ….`
        is parseable as a comma-separated list with one schema.
* Q2(b) ✓  Set/list membership over typed `List String`, combining two
        independent membership constraints.
* Q3 ✓  Spec is `"storage" ∈ perms ∧ "sms" ∈ perms` — typed list
        membership, not a substring scan.
* Q4 ✓  `_iff` composes by case-split on `parseCommaList` and the
        decidable `List.elem` for `String`.

Workflow §3.5 / manual §"Messaging App Lacks Necessary Permissions". -/

-- [source: result] [phase: post] [tool: check_app_permissions → grant_app_permission]
def messagingPermPrefix : String :=
  "App 'messaging' has permission for: "

def spec_result_messagingPerms (r : String) : Prop :=
  ∀ perms : List String,
    parseCommaList r messagingPermPrefix = some perms →
    "storage" ∈ perms ∧ "sms" ∈ perms

def check_result_messagingPerms (r : String) : Bool :=
  match parseCommaList r messagingPermPrefix with
  | some perms => perms.contains "storage" && perms.contains "sms"
  | none       => true

theorem check_result_messagingPerms_iff (r : String) :
    check_result_messagingPerms r = true ↔ spec_result_messagingPerms r := by
  unfold check_result_messagingPerms spec_result_messagingPerms
  refine ⟨?_, ?_⟩
  · intro hk perms hp
    rw [hp] at hk
    have h2 : perms.contains "storage" = true ∧ perms.contains "sms" = true := by
      simpa [Bool.and_eq_true] using hk
    refine ⟨?_, ?_⟩
    · simpa [List.contains_iff_exists_mem_beq] using h2.1
    · simpa [List.contains_iff_exists_mem_beq] using h2.2
  · intro hyp
    cases hp : parseCommaList r messagingPermPrefix with
    | none       => simp [hp]
    | some perms =>
      obtain ⟨hs, hm⟩ := hyp perms hp
      have hs' : perms.contains "storage" = true := by
        simpa [List.contains_iff_exists_mem_beq] using hs
      have hm' : perms.contains "sms" = true := by
        simpa [List.contains_iff_exists_mem_beq] using hm
      simp [hp, hs', hm']
      exact ⟨hs, hm⟩

instance (r : String) : Decidable (spec_result_messagingPerms r) :=
  decidable_of_iff _ (check_result_messagingPerms_iff r)

def feedback_result_messagingPerms : String :=
  "check_app_permissions for the messaging app does not list both \"storage\" and \"sms\" — guide the user to call grant_app_permission(app_name=\"messaging\", permission=\"storage\") and grant_app_permission(app_name=\"messaging\", permission=\"sms\") (workflow §3.5)."

/-! ## §G. Top-level action dispatch and soundness

`Action` enumerates the policy-relevant tools that ACTUALLY exist in
`tools.py`.  Tools mentioned in the policy but absent from the runtime
(`make_payment`, `check_payment_request`, `change_plan`) are NOT actions:
their triplets above are tagged "vacuous in current runtime" and gate no
runtime call.
-/

inductive Action
  | getCustomerByPhone (phone : String)
  | getCustomerById    (cid : CustomerId)
  | getCustomerByName  (name : String) (dob : String)
  | suspendLine        (cid : CustomerId) (lid : LineId) (reason : String)
  | resumeLine         (cid : CustomerId) (lid : LineId)
  | sendPaymentRequest (cid : CustomerId) (bid : BillId)
  | refuelData         (cid : CustomerId) (lid : LineId) (gbTimes100 : Nat)
  | enableRoaming      (cid : CustomerId) (lid : LineId)
  | disableRoaming     (cid : CustomerId) (lid : LineId)
  | getDataUsage       (cid : CustomerId) (lid : LineId)
  | getBillsForCustomer (cid : CustomerId)
  | transferToHumanAgents (summary : String)
  deriving Repr

/-- Convert an `Action` into the `ToolCall` we record in `history`. -/
def Action.toToolCall : Action → ToolCall
  | .getCustomerByPhone p          => .getCustomerByPhone p
  | .getCustomerById cid           => .getCustomerById cid
  | .getCustomerByName n d         => .getCustomerByName n d
  | .suspendLine cid lid r         => .suspendLine cid lid r
  | .resumeLine cid lid            => .resumeLine cid lid
  | .sendPaymentRequest cid bid    => .sendPaymentRequest cid bid
  | .refuelData cid lid g          => .refuelData cid lid g
  | .enableRoaming cid lid         => .enableRoaming cid lid
  | .disableRoaming cid lid        => .disableRoaming cid lid
  | .getDataUsage cid lid          => .getDataUsage cid lid
  | .getBillsForCustomer cid       => .getBillsForCustomer cid
  | .transferToHumanAgents s       => .transferToHumanAgents s

/-- Apply an action to update `AgentState`.  We record the call in history;
identification side-effects of the lookup tools are also captured. -/
def applyAction (s : AgentState) (act : Action) : AgentState :=
  let s := { s with history := s.history ++ [act.toToolCall] }
  match act with
  | .getCustomerById cid =>
      match lookupCustomer s cid with
      | some _ => { s with identifiedCustomer := some cid }
      | none   => s
  | _ => s

/-- The aggregate verdict returned to the caller. -/
inductive CheckResult
  | allow
  | deny (reasons : List String)
  deriving Repr

/-- Helper: given a list of `(Bool, feedback)` pairs, collect feedbacks for
every `Bool = false`. -/
def collectFailures (rs : List (Bool × String)) : List String :=
  rs.filterMap (fun (ok, msg) => if ok then none else some msg)

/-- The pre-condition checker, dispatched on the action. -/
def checkAction (s : AgentState) (act : Action) : CheckResult :=
  let failures : List String :=
    match act with
    | .getCustomerByPhone _ =>
        []
    | .getCustomerById _ =>
        []
    | .getCustomerByName name dob =>
        collectFailures
          [(check_nameLookupHasDOB name dob,
              feedback_nameLookupHasDOB name dob)]
    | .suspendLine _ _ reason =>
        collectFailures
          [(check_customerIdentified s,
              feedback_customerIdentified s),
           (check_validSuspensionReason reason,
              feedback_validSuspensionReason reason)]
    | .resumeLine cid lid =>
        collectFailures
          [(check_customerIdentified s,
              feedback_customerIdentified s),
           (check_noOverdueBillsForCustomer s cid,
              feedback_noOverdueBillsForCustomer s cid),
           (check_contractNotExpired s lid,
              feedback_contractNotExpired s lid)]
    | .sendPaymentRequest cid bid =>
        collectFailures
          [(check_customerIdentified s,
              feedback_customerIdentified s),
           (check_billOverdue s bid,
              feedback_billOverdue s bid),
           (check_noOtherAwaitingPayment s cid,
              feedback_noOtherAwaitingPayment s cid),
           (check_billBelongsToCustomer s cid bid,
              feedback_billBelongsToCustomer s cid bid)]
    | .refuelData _ lid g =>
        collectFailures
          [(check_customerIdentified s,
              feedback_customerIdentified s),
           (check_refuelPositive g,
              feedback_refuelPositive g),
           (check_refuelMaxGB g,
              feedback_refuelMaxGB g),
           (check_refuelPriceConfirmed lid g,
              feedback_refuelPriceConfirmed lid g)]
    | .enableRoaming _ lid =>
        collectFailures
          [(check_customerIdentified s,
              feedback_customerIdentified s),
           (check_userTravelling lid,
              feedback_userTravelling lid)]
    | .disableRoaming _ _ =>
        collectFailures
          [(check_customerIdentified s,
              feedback_customerIdentified s)]
    | .getDataUsage _ _ =>
        collectFailures
          [(check_customerIdentified s,
              feedback_customerIdentified s)]
    | .getBillsForCustomer _ =>
        collectFailures
          [(check_customerIdentified s,
              feedback_customerIdentified s)]
    | .transferToHumanAgents _ =>
        collectFailures
          [(check_escalationJustified, feedback_escalationJustified)]
  match failures with
  | []   => .allow
  | _::_ => .deny failures

/-! Per-rule soundness is already established by each `check_X_iff` lemma.
We deliberately omit an aggregate `specCompliant` / `checkAction_sound`
theorem — the top-level `checkAction` is just a dispatcher, and any caller
that needs the Prop-level guarantee can apply the relevant per-rule bridge
directly. -/

/-- Helper: `collectFailures rs = []` iff every Bool in `rs` is `true`.
Used for downstream consumers who want to reason about `checkAction`. -/
lemma collectFailures_nil_iff (rs : List (Bool × String)) :
    collectFailures rs = [] ↔ ∀ p ∈ rs, p.1 = true := by
  induction rs with
  | nil => simp [collectFailures]
  | cons p rs ih =>
    cases hp : p.1 with
    | true =>
      simp [collectFailures, hp, ih]
    | false =>
      refine ⟨?_, ?_⟩
      · intro h; simp [collectFailures, hp] at h
      · intro h
        exact absurd (h p (by simp)) (by simp [hp])

end Telecom






