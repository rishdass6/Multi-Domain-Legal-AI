GOLDEN_DATASET = [
    # Confidentiality
    {"id": "conf_001", "question": "How long does the confidentiality obligation last?",
     "keywords": ["year", "years", "months", "period", "survive"]},
    {"id": "conf_002", "question": "What information is deemed confidential under this agreement?",
     "keywords": ["confidential", "proprietary", "information", "trade secret"]},
    {"id": "conf_003", "question": "What are the exceptions to the confidentiality obligation?",
     "keywords": ["public", "domain", "required", "law", "prior", "known"]},

    # Governing Law
    {"id": "gov_001", "question": "The laws of which state shall govern this agreement?",
     "keywords": ["delaware", "new york", "california", "texas", "state", "illinois"]},
    {"id": "gov_002", "question": "Which courts have jurisdiction over disputes?",
     "keywords": ["court", "jurisdiction", "county", "district", "state", "federal"]},

    # Termination
    {"id": "term_001", "question": "How many days written notice is required to terminate?",
     "keywords": ["30", "60", "90", "days", "notice"]},
    {"id": "term_002", "question": "Under what conditions may a party terminate for cause?",
     "keywords": ["breach", "default", "material", "cure", "fail"]},
    {"id": "term_003", "question": "What is the initial term of this agreement?",
     "keywords": ["year", "years", "months", "term", "period"]},
    {"id": "term_004", "question": "What happens to outstanding payments upon termination?",
     "keywords": ["payment", "fees", "accrued", "due", "outstanding", "survive"]},

    # Limitation of Liability
    {"id": "liab_001", "question": "Are indirect or consequential damages excluded?",
     "keywords": ["indirect", "consequential", "incidental", "excluded", "liable", "special"]},
    {"id": "liab_002", "question": "What is the maximum aggregate liability cap?",
     "keywords": ["aggregate", "exceed", "fees", "paid", "months", "total", "cap"]},

    # Indemnification
    {"id": "indem_001", "question": "Who shall indemnify and hold harmless the other party?",
     "keywords": ["indemnif", "licensee", "licensor", "company", "party", "shall"]},
    {"id": "indem_002", "question": "What third party claims must be indemnified?",
     "keywords": ["claim", "suit", "action", "loss", "damage", "expense", "cost"]},
    {"id": "indem_003", "question": "What notice must be given to trigger indemnification?",
     "keywords": ["notice", "written", "prompt", "notify", "days"]},

    # Intellectual Property
    {"id": "ip_001", "question": "Who owns intellectual property created under this agreement?",
     "keywords": ["own", "sole", "exclusive", "assign", "company", "licensor", "work"]},
    {"id": "ip_002", "question": "What license rights are granted to the licensee?",
     "keywords": ["non-exclusive", "royalty", "license", "right", "use", "sublicense"]},
    {"id": "ip_003", "question": "Are improvements to the licensed technology owned by the licensor?",
     "keywords": ["improvement", "enhancement", "licensor", "own", "assign", "vest"]},

    # Force Majeure
    {"id": "fm_001", "question": "What constitutes a force majeure event under this agreement?",
     "keywords": ["war", "flood", "fire", "act of god", "strike", "beyond", "control", "natural"]},
    {"id": "fm_002", "question": "How long can a force majeure event suspend performance?",
     "keywords": ["days", "months", "period", "suspend", "duration", "notice"]},

    # Payment
    {"id": "pay_001", "question": "When are invoices due under this agreement?",
     "keywords": ["days", "net", "receipt", "invoice", "due", "30", "60"]},
    {"id": "pay_002", "question": "What interest applies to late payments?",
     "keywords": ["interest", "percent", "rate", "late", "overdue", "per annum"]},

    # Dispute Resolution
    {"id": "disp_001", "question": "How shall disputes between the parties be resolved?",
     "keywords": ["arbitrat", "mediat", "court", "litigation", "proceeding", "resolv"]},
    {"id": "disp_002", "question": "Where shall arbitration proceedings take place?",
     "keywords": ["new york", "london", "arbitration", "venue", "seat", "location", "city"]},

    # Assignment
    {"id": "assign_001", "question": "Can either party assign this agreement without consent?",
     "keywords": ["assign", "consent", "prior", "written", "approval", "transfer"]},
    {"id": "assign_002", "question": "Is assignment permitted in connection with a merger?",
     "keywords": ["merger", "acquisition", "change of control", "assign", "successor"]},

    # Warranties
    {"id": "warr_001", "question": "What warranties does the licensor provide?",
     "keywords": ["warrant", "represent", "infringe", "title", "right", "authority"]},
    {"id": "warr_002", "question": "Is there a disclaimer of implied warranties?",
     "keywords": ["disclaim", "implied", "merchantab", "fitness", "warranty", "as is"]},

    # Non-Compete
    {"id": "nc_001", "question": "What business activities is the party restricted from?",
     "keywords": ["compet", "business", "engage", "restrict", "prohibit", "solicit"]},
    {"id": "nc_002", "question": "How long does the non-compete restriction last?",
     "keywords": ["year", "years", "months", "period", "term", "duration"]},

    # General
    {"id": "gen_001", "question": "Can this agreement be amended verbally?",
     "keywords": ["written", "writing", "amendment", "signed", "parties", "modify"]},
]