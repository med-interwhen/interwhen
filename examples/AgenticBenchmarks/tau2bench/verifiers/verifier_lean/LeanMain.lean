/-
  LeanMain.lean

  Long-lived stdin/stdout policy-check daemon for the telecom domain.

  Scope: only the 9 DB-driven proven checks from PolicyChecker.lean.
  Everything else (POST coaching, empirical arg-matching, hypothesis
  gates) is handled in Python (telecom_python_rules.py) — Lean adds no
  proof-checked value to those, and putting them here was bloat.

  Protocol (line-delimited JSON):

    request:
      { "id":          "<echoed>",
        "rule":        "<rule_id>",
        "tool_args":   { ... },
        "db_snapshot": { "today": <int>, "identified_customer": "<cid|null>",
                         "customers": [...], "lines": [...],
                         "bills": [...] } }

    response (success):
      { "id":"<echoed>", "ok": true, "verdict": null | "<feedback string>" }

    response (error):
      { "id":"<echoed-or-null>", "ok": false, "error": "<message>" }

  Adding a new rule:
    1. Define `check_X` / `feedback_X` in PolicyChecker.lean.
    2. Append one entry to `ruleTable` below — that's it.
-/
import PolicyChecker
import Lean.Data.Json

open Lean Telecom

namespace PolicyDaemon

/-! ## JSON helpers -/

def getStr  (j : Json) (k : String) : String :=
  (j.getObjValAs? String k).toOption.getD ""

def getNat  (j : Json) (k : String) : Nat :=
  (j.getObjValAs? Nat k).toOption.getD 0

def getInt  (j : Json) (k : String) : Int :=
  (j.getObjValAs? Int k).toOption.getD 0

def getBool (j : Json) (k : String) : Bool :=
  (j.getObjValAs? Bool k).toOption.getD false

def getArr  (j : Json) (k : String) : Array Json :=
  match j.getObjVal? k with
  | .ok (Json.arr xs) => xs
  | _                 => #[]

def getObj? (j : Json) (k : String) : Option Json :=
  (j.getObjVal? k).toOption

def jsonStr (j : Json) : String :=
  match j with | .str s => s | _ => ""

/-! ## ID constructors and enum parsing -/

def mkCid (s : String) : CustomerId := ⟨s⟩
def mkLid (s : String) : LineId     := ⟨s⟩
def mkBid (s : String) : BillId     := ⟨s⟩
def mkPid (s : String) : PlanId     := ⟨s⟩

def parseBillStatus : String → BillStatus
  | "Draft"            => .draft
  | "Issued"           => .issued
  | "Paid"             => .paid
  | "Overdue"          => .overdue
  | "Awaiting Payment" => .awaitingPayment
  | "Disputed"         => .disputed
  | _                  => .issued

def parseLineStatus : String → LineStatus
  | "Active"             => .active
  | "Suspended"          => .suspended
  | "Pending Activation" => .pendingActivation
  | _                    => .closed

def parseAccountStatus : String → AccountStatus
  | "Active"               => .active
  | "Suspended"            => .suspended
  | "Pending Verification" => .pendingVerification
  | _                      => .closed

/-! ## Record builders -/

def parseBill (j : Json) : Bill :=
  { billId     := mkBid (getStr j "bill_id")
    customerId := mkCid (getStr j "customer_id")
    totalDue   := getNat j "total_due"
    status     := parseBillStatus (getStr j "status") }

def parseLine (j : Json) : Line :=
  { lineId          := mkLid (getStr j "line_id")
    phoneNumber     := getStr j "phone_number"
    status          := parseLineStatus (getStr j "status")
    planId          := mkPid (getStr j "plan_id")
    ownerId         := mkCid (getStr j "owner_id")
    dataUsedGb      := getNat j "data_used_gb"
    dataRefuelingGb := getNat j "data_refueling_gb"
    roamingEnabled  := getBool j "roaming_enabled"
    contractEndDate := getInt j "contract_end_date" }

def parseCustomer (j : Json) : Customer :=
  let st :=
    let a := getStr j "account_status"
    if a ≠ "" then a else getStr j "status"
  { customerId  := mkCid (getStr j "customer_id")
    fullName    := getStr j "full_name"
    dateOfBirth := getStr j "date_of_birth"
    email       := getStr j "email"
    phoneNumber := getStr j "phone_number"
    status      := parseAccountStatus st
    lineIds     := (getArr j "line_ids").toList.map (fun x => mkLid (jsonStr x))
    billIds     := (getArr j "bill_ids").toList.map (fun x => mkBid (jsonStr x)) }

def parseAgentState (snap : Json) : AgentState :=
  let identified : Option CustomerId :=
    match snap.getObjVal? "identified_customer" with
    | .ok (.str s) => if s = "" then none else some (mkCid s)
    | _            => none
  { customers          := (getArr snap "customers").toList.map parseCustomer
    bills              := (getArr snap "bills").toList.map parseBill
    lines              := (getArr snap "lines").toList.map parseLine
    plans              := []
    identifiedCustomer := identified
    history            := []
    today              := getInt snap "today"
    lastToolResults    := []
    userPhone          := getStr snap "user_phone" }

/-! ## Rule registry

Every rule is a closure `(state, args) -> Option String`. Adding a new
rule means appending one entry below; `runRule` itself is a one-liner. -/

abbrev Rule := AgentState → Json → Option String

def ruleTable : List (String × Rule) := [
  ("check_nameLookupHasDOB", fun _ a =>
    let n := getStr a "full_name"
    let d := getStr a "dob"
    if check_nameLookupHasDOB n d then none
    else some (feedback_nameLookupHasDOB n d)),

  ("check_customerIdentified", fun s _ =>
    if check_customerIdentified s then none
    else some (feedback_customerIdentified s)),

  ("check_billOverdue", fun s a =>
    let b := mkBid (getStr a "bill_id")
    if check_billOverdue s b then none
    else some (feedback_billOverdue s b)),

  ("check_noOtherAwaitingPayment", fun s a =>
    let c := mkCid (getStr a "customer_id")
    if check_noOtherAwaitingPayment s c then none
    else some (feedback_noOtherAwaitingPayment s c)),

  ("check_billBelongsToCustomer", fun s a =>
    let c := mkCid (getStr a "customer_id")
    let b := mkBid (getStr a "bill_id")
    if check_billBelongsToCustomer s c b then none
    else some (feedback_billBelongsToCustomer s c b)),

  ("check_noOverdueBillsForCustomer", fun s a =>
    let c := mkCid (getStr a "customer_id")
    if check_noOverdueBillsForCustomer s c then none
    else some (feedback_noOverdueBillsForCustomer s c)),

  ("check_contractNotExpired", fun s a =>
    let l := mkLid (getStr a "line_id")
    if check_contractNotExpired s l then none
    else some (feedback_contractNotExpired s l)),

  ("check_refuelMaxGB", fun _ a =>
    let g := getNat a "gb_times_100"
    if check_refuelMaxGB g then none
    else some (feedback_refuelMaxGB g)),

  ("check_refuelPositive", fun _ a =>
    let g := getNat a "gb_times_100"
    if check_refuelPositive g then none
    else some (feedback_refuelPositive g)),

  -- POST checks. `tool_args` carries `result : String` (raw tool output).
  ("check_result_dataUsage_exceeded", fun _ a =>
    let r := getStr a "result"
    if check_result_dataUsage_exceeded r then none
    else some feedback_result_dataUsage_exceeded),

  ("check_result_linePhoneMatchesState", fun s a =>
    let r := getStr a "result"
    if check_result_linePhoneMatchesState s r then none
    else some feedback_result_linePhoneMatchesState),

  ("check_result_messagingPerms", fun _ a =>
    let r := getStr a "result"
    if check_result_messagingPerms r then none
    else some feedback_result_messagingPerms)
]

def runRule (rule : String) (s : AgentState) (args : Json) : Option String :=
  (ruleTable.lookup rule).bind (fun f => f s args)

/-! ## Request / response -/

def mkResponse (id : Json) (verdict : Option String) : Json :=
  Json.mkObj [
    ("id",      id),
    ("ok",      .bool true),
    ("verdict", verdict.map Json.str |>.getD .null)
  ]

def mkErrorResponse (id : Json) (err : String) : Json :=
  Json.mkObj [
    ("id",    id),
    ("ok",    .bool false),
    ("error", .str err)
  ]

def handleRequest (j : Json) : Json :=
  let id    := (j.getObjVal? "id").toOption.getD .null
  let rule  := getStr j "rule"
  let args  := (getObj? j "tool_args").getD (Json.mkObj [])
  let snap  := (getObj? j "db_snapshot").getD (Json.mkObj [])
  mkResponse id (runRule rule (parseAgentState snap) args)

partial def loop : IO Unit := do
  let stdin  ← IO.getStdin
  let stdout ← IO.getStdout
  let stderr ← IO.getStderr
  let rec go : IO Unit := do
    let line ← stdin.getLine
    if line.isEmpty then return  -- EOF
    let trimmed := line.trim
    if trimmed.isEmpty then go else
    match Json.parse trimmed with
    | .error e =>
        stdout.putStr (Json.compress (mkErrorResponse .null s!"json parse error: {e}") ++ "\n")
        stdout.flush
        go
    | .ok j =>
        if (j.getObjValAs? Bool "shutdown").toOption == some true then return
        stdout.putStr (Json.compress (handleRequest j) ++ "\n")
        stdout.flush
        go
  go
  stderr.putStr "policy-daemon: shutting down\n"

end PolicyDaemon

def main : IO Unit := PolicyDaemon.loop
