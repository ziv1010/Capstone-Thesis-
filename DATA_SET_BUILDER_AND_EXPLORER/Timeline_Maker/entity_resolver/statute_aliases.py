"""
Curated statute alias dictionary built from the authority-degree CSVs.

Each entry maps a canonical_id -> (display_name, [alias regex patterns]).
Patterns match against the LOWERCASED+WHITESPACE-NORMALIZED+PUNCT-STRIPPED form
of the raw entity text. They are anchored with `^...$` automatically.

Order matters: patterns are tried in dictionary order and the first match wins.
List the more specific canonicals BEFORE more generic ones (e.g. RFCTLARR before
generic Land Acquisition Act, since RFCTLARR strings often contain "land
acquisition").
"""
from __future__ import annotations

import re
import string

WHITESPACE_RE = re.compile(r"\s+")
PUNCT_TRANSLATION = str.maketrans(string.punctuation, " " * len(string.punctuation))
# Glue letters separated by a dot ("Cr.P.C." -> "CrPC"); apply repeatedly.
DOT_BETWEEN_LETTERS = re.compile(r"([A-Za-z0-9])\.([A-Za-z0-9])")
# Match a run of 2+ single-letter tokens separated by whitespace ("d v", "i p c").
SINGLE_LETTER_RUN = re.compile(r"\b(?:[a-z]\s+){1,}[a-z]\b")


def _glue_dot_letters(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = DOT_BETWEEN_LETTERS.sub(r"\1\2", s)
    return s


def _collapse_single_letters(s: str) -> str:
    return SINGLE_LETTER_RUN.sub(lambda m: m.group(0).replace(" ", ""), s)


def normalize_for_alias(text: str) -> str:
    s = text.lower()
    s = _glue_dot_letters(s)
    s = s.translate(PUNCT_TRANSLATION)
    s = WHITESPACE_RE.sub(" ", s).strip()
    s = _collapse_single_letters(s)
    s = WHITESPACE_RE.sub(" ", s).strip()
    return s


# canonical_id -> (display_name, [regex_patterns])
# IMPORTANT: ordering matters. More specific (longer) statutes come first.
STATUTE_CANONICALS: list[tuple[str, str, list[str]]] = [
    # ----- Land Acquisition family (specific RFCTLARR before generic LA Act) -----
    (
        "rfctlarr_2013",
        "Right to Fair Compensation and Transparency in Land Acquisition, Rehabilitation and Resettlement Act, 2013",
        [
            r"right to fair compensation.*2013.*",
            r"right of fair compensation.*2013.*",
            r"compensation and transparency in land acquisition.*2013.*",
            r"land acquisition rehabilitation and resettlement act 2013",
            r"rehabilitation and resettlement act 2013",
            r"rfctlarr.*",
            r"larr 2013 act",
            r"2013 act",
            r"act 2013",
            r"act 30 of 2013",
            r"la act of 2013",
            r"new land acquisition act",
            r"aofresaid act of 2013",
            r"fair compensation act",
        ],
    ),
    (
        "land_acquisition_act_1894",
        "Land Acquisition Act, 1894",
        [
            r"land acquisition act 1894.*",
            r"land of acquisition act 1894",
            r"land acquisition act of 1894",
            r"land acquisition act 1984",  # observed typo for 1894 in CSV
            r"la act 1894",
            r"la act of 1894",
            r"l a act",
            r"la act",
            r"land acquisition act",
            r"land acquisition amendment act 1984.*",
            r"act 1894",
            r"act of 1894",
            r"1894 act",
            r"act no 1 of 1894",
            r"act i of 1894",
            r"acquisition act",
            r"groupc doc acquisition act 1894",
        ],
    ),
    # ----- Indian Penal Code -----
    (
        "ipc",
        "Indian Penal Code, 1860",
        [
            r"ipc.*",
            r"indian penal code.*",
            r"penal code.*",
            r"i p c",
        ],
    ),
    # ----- BNSS (replaced CrPC in 2023) - before BNS so 'bnss' isn't gobbled by 'bns.*' -----
    (
        "bnss_2023",
        "Bharatiya Nagarik Suraksha Sanhita, 2023",
        [
            r"bharatiya nagarik suraksha sanhita.*",
            r"bnss.*",
        ],
    ),
    # ----- BNS (replaced IPC in 2023) -----
    (
        "bns_2023",
        "Bharatiya Nyaya Sanhita, 2023",
        [
            r"bharatiya nyaya sanhita.*",
            r"bhartiya nyay.*sanhita.*",  # observed misspelling
            r"bhartiya nyaya sanhita.*",
            r"bns.*",
        ],
    ),
    # ----- Code of Criminal Procedure -----
    (
        "crpc",
        "Code of Criminal Procedure, 1973",
        [
            r"code of criminal procedure.*",
            r"criminal procedure code.*",
            r"crpc.*",
            r"cr pc",
            r"cr p c",
            r"cr",
        ],
    ),
    # ----- Code of Civil Procedure -----
    (
        "cpc",
        "Code of Civil Procedure, 1908",
        [
            r"code of civil procedure.*",
            r"civil procedure code.*",
            r"cpc.*",
            r"cpcand",
        ],
    ),
    # ----- Constitution of India -----
    (
        "constitution_of_india",
        "Constitution of India",
        [
            r"constitution of india.*",
            r"indian constitution",
            r"constitution",
            r"constitution of the",
            r"constitution of of natural",
            r"indian act",  # ambiguous but seen in CSV in constitution-cluster context (low count)
        ],
    ),
    # ----- Motor Vehicles Act -----
    (
        "motor_vehicles_act_1988",
        "Motor Vehicles Act, 1988",
        [
            r"motor vehicles act 1988.*",
            r"motor vehicle act 1988.*",
            r"mv act 1988.*",
            r"mvact 1988.*",
            r"motor vehicles act",
            r"motor vehicle act",
            r"mv act",
            r"m v act",
            r"mvact",
            r"imv act",
            r"motor vehicles act1988 the",
            r"motor vehicles act 1988the",
            r"motor vehicles amendment act 2019.*",
            r"vehicles act",
        ],
    ),
    (
        "motor_vehicles_act_1939",
        "Motor Vehicles Act, 1939",
        [r"motor vehicles act 1939"],
    ),
    # ----- Evidence Act -----
    (
        "indian_evidence_act_1872",
        "Indian Evidence Act, 1872",
        [
            r"indian evidence act.*",
            r"evidence act.*",
            r"iea",
        ],
    ),
    # ----- POCSO -----
    (
        "pocso_2012",
        "Protection of Children from Sexual Offences Act, 2012",
        [
            r"protection of children from sexual offences act.*",
            r"pocso act.*",
            r"pocso rules.*",
            r"pocso.*",
            r"posco act",  # observed typo
        ],
    ),
    # ----- National Highways Act -----
    (
        "national_highways_act_1956",
        "National Highways Act, 1956",
        [
            r"national highways? act.*",
            r"national highway act.*",
            r"nh act.*",
            r"nhact",
            r"highways act",
        ],
    ),
    (
        "nhai_act_1988",
        "National Highways Authority of India Act, 1988",
        [
            r"national highways authority.*1988.*",
            r"national highways authority act",
            r"nhai act",
        ],
    ),
    # ----- Dowry Prohibition Act -----
    (
        "dowry_prohibition_act_1961",
        "Dowry Prohibition Act, 1961",
        [
            r"dowry prohibition act.*",
            r"dowry prohibiton act.*",  # observed typo
            r"prohibition of dowry act.*",
            r"dowry act",
            r"dp act.*",
            r"d p act",
            r"dpact.*",
        ],
    ),
    # ----- Domestic Violence Act -----
    (
        "pwdv_2005",
        "Protection of Women from Domestic Violence Act, 2005",
        [
            r"protection of women from domestic violence act.*",
            r"prevention of domestic violence act.*",
            r"domestic violence act.*",
            r"dv act.*",
            r"dvact.*",
            r"pwdv act",
            r"pwdvact",
        ],
    ),
    # ----- Hindu Marriage Act -----
    (
        "hindu_marriage_act_1955",
        "Hindu Marriage Act, 1955",
        [
            r"hindu marriage act.*",
            r"hma",
        ],
    ),
    # ----- Family Courts Act -----
    (
        "family_courts_act_1984",
        "Family Courts Act, 1984",
        [
            r"family courts? act.*",
        ],
    ),
    # ----- Guardians and Wards Act -----
    (
        "guardians_and_wards_act_1890",
        "Guardians and Wards Act, 1890",
        [
            r"guardians and wards act.*",
            r"gwact.*",
            r"gw act.*",
        ],
    ),
    # ----- Senior Citizens Act -----
    (
        "senior_citizens_act_2007",
        "Maintenance and Welfare of Parents and Senior Citizens Act, 2007",
        [
            r"maintenance.*welfare of parents.*senior citizens act.*",
            r"senior citizens? act.*",
        ],
    ),
    # ----- Hindu Succession Act -----
    (
        "hindu_succession_act_1956",
        "Hindu Succession Act, 1956",
        [r"hindu succession act.*"],
    ),
    # ----- Probation of Offenders Act -----
    (
        "probation_of_offenders_act_1958",
        "Probation of Offenders Act, 1958",
        [
            r"probation of offenders act.*",
            r"probation act",
            r"po act",
            r"uttar pradesh first offenders probation act 1938",  # state variant — keep separate? group for now
        ],
    ),
    # ----- Arbitration and Conciliation Act -----
    (
        "arbitration_and_conciliation_act_1996",
        "Arbitration and Conciliation Act, 1996",
        [
            r"arbitration and conciliation act.*",
            r"arbitration conciliation act.*",
            r"arbitration act.*",
            r"act 1996",
        ],
    ),
    # ----- Arms Act -----
    (
        "arms_act_1959",
        "Arms Act, 1959",
        [r"arms act.*"],
    ),
    # ----- Limitation Act -----
    (
        "limitation_act_1963",
        "Limitation Act, 1963",
        [
            r"limitation act.*",
            r"indian limitation act of 1963",
        ],
    ),
    # ----- Indian Contract Act -----
    (
        "indian_contract_act_1872",
        "Indian Contract Act, 1872",
        [
            r"indian contract act.*",
            r"contract act",
        ],
    ),
    # ----- Transfer of Property Act -----
    (
        "transfer_of_property_act_1882",
        "Transfer of Property Act, 1882",
        [r"transfer of property act.*"],
    ),
    # ----- Specific Relief Act -----
    (
        "specific_relief_act_1963",
        "Specific Relief Act, 1963",
        [
            r"specific relief act.*",
            r"sra",
        ],
    ),
    # ----- Indian Stamp Act -----
    (
        "indian_stamp_act_1899",
        "Indian Stamp Act, 1899",
        [
            r"indian stamp act.*",
            r"stamp act",
        ],
    ),
    # ----- Registration Act -----
    (
        "registration_act_1908",
        "Registration Act, 1908",
        [r"registration act.*"],
    ),
    # ----- Negotiable Instruments Act -----
    (
        "ni_act_1881",
        "Negotiable Instruments Act, 1881",
        [
            r"negotiable instruments? act.*",
            r"ni act.*",
            r"niact",
        ],
    ),
    # ----- Prevention of Corruption Act (1988) -----
    (
        "pc_act_1988",
        "Prevention of Corruption Act, 1988",
        [
            r"prevention of corruption act 1988.*",
            r"prevention of corruption act",
            r"pc act.*",
            r"pcact",
            r"pc",
        ],
    ),
    (
        "pc_act_1947",
        "Prevention of Corruption Act, 1947",
        [r"prevention of corruption act 1947"],
    ),
    # ----- PMLA -----
    (
        "pmla_2002",
        "Prevention of Money Laundering Act, 2002",
        [
            r"prevention of money laundering act.*",
            r"pmla.*",
            r"pml act.*",
        ],
    ),
    # ----- IT Act -----
    (
        "it_act_2000",
        "Information Technology Act, 2000",
        [
            r"information technology act.*",
            r"it act.*",
        ],
    ),
    # ----- Income Tax Act -----
    (
        "income_tax_act_1961",
        "Income Tax Act, 1961",
        [
            r"income tax act 1961.*",
            r"income tax act",
            r"incometax act.*",
            r"indian income tax act",
            r"central income tax act",
        ],
    ),
    (
        "income_tax_act_1922",
        "Income Tax Act, 1922",
        [r"income tax act 1922"],
    ),
    # ----- General Clauses Act -----
    (
        "general_clauses_act_1897",
        "General Clauses Act, 1897",
        [
            r"general clauses act.*",
            r"gc act",
        ],
    ),
    # ----- Workmen's / Employees' Compensation Act -----
    (
        "employees_compensation_act_1923",
        "Employee's Compensation Act, 1923",  # was Workmen's Compensation Act, 1923, renamed in 2009
        [
            r"workmens compensation act.*",
            r"workmen compensation act.*",
            r"employees compensation act.*",
        ],
    ),
    # ----- Minimum Wages Act -----
    (
        "minimum_wages_act_1948",
        "Minimum Wages Act, 1948",
        [r"minimum wages act.*"],
    ),
    # ----- Industrial Disputes Act -----
    (
        "industrial_disputes_act_1947",
        "Industrial Disputes Act, 1947",
        [r"industrial disputes? act.*"],
    ),
    # ----- Juvenile Justice Act -----
    (
        "jj_act_2015",
        "Juvenile Justice (Care and Protection of Children) Act, 2015",
        [
            r"juvenile justice care and protection of children act 2015.*",
            r"juvenile justice care and protection of children act",
            r"juvenile justice act.*",
            r"jj act 2015",
            r"jj act",
            r"juvenile justice care and protection of children rules.*",
            r"jj rules.*",
            r"juvenile justice rules",
        ],
    ),
    (
        "jj_act_2000",
        "Juvenile Justice (Care and Protection of Children) Act, 2000",
        [r"juvenile justice care and protection of children act 2000"],
    ),
    (
        "children_act_1960",
        "Children Act, 1960",
        [r"children act 1960.*"],
    ),
    # ----- SC/ST Atrocities Act -----
    (
        "scst_poa_1989",
        "Scheduled Castes and Scheduled Tribes (Prevention of Atrocities) Act, 1989",
        [
            r"scheduled castes and scheduled tribes prevention of atrocities act.*",
            r"scheduled castes and the scheduled tribes prevention of atrocities act.*",
            r"scst act",
            r"sc st act",
            r"scst poa act",
            r"atrocities act",
        ],
    ),
    # ----- NDPS Act -----
    (
        "ndps_act_1985",
        "Narcotic Drugs and Psychotropic Substances Act, 1985",
        [
            r"narcotic drugs and psychotropic substances act.*",
            r"ndps act.*",
        ],
    ),
    # ----- UAPA -----
    (
        "uapa_1967",
        "Unlawful Activities (Prevention) Act, 1967",
        [
            r"unlawful activities.*prevention.*act.*",
            r"uapa.*",
            r"uap act",
        ],
    ),
    # ----- TADA -----
    (
        "tada_1987",
        "Terrorist and Disruptive Activities (Prevention) Act, 1987",
        [
            r"terrorist and disruptive activities.*",
            r"tada",
        ],
    ),
    # ----- Right to Information Act -----
    (
        "rti_act_2005",
        "Right to Information Act, 2005",
        [
            r"right to information act.*",
            r"rti act.*",
        ],
    ),
    # ----- SARFAESI -----
    (
        "sarfaesi_2002",
        "Securitisation and Reconstruction of Financial Assets and Enforcement of Security Interest Act, 2002",
        [
            r"sarfaesi act.*",
            r"security interest enforcement rules.*",
        ],
    ),
    # ----- Customs Act -----
    (
        "customs_act_1962",
        "Customs Act, 1962",
        [
            r"customs act.*",
            r"customs amendment act.*",
        ],
    ),
    # ----- Central Excise Act -----
    (
        "central_excise_act_1944",
        "Central Excise Act, 1944",
        [
            r"central excise act.*",
            r"excise act.*",
        ],
    ),
    # ----- GST -----
    (
        "cgst_2017",
        "Central Goods and Services Tax Act, 2017",
        [
            r"central goods and services tax act.*",
            r"cgst.*",
            r"gst",
        ],
    ),
    # ----- Companies Act -----
    (
        "companies_act_2013",
        "Companies Act, 2013",
        [r"companies act 2013.*"],
    ),
    (
        "companies_act_1956",
        "Companies Act, 1956",
        [r"companies act 1956.*"],
    ),
    (
        "companies_act_generic",
        "Companies Act",
        [r"companies act"],  # fallback when year not specified
    ),
    # ----- Indian Telegraph Act -----
    (
        "indian_telegraph_act_1885",
        "Indian Telegraph Act, 1885",
        [
            r"indian telegraph act.*",
            r"telegraph act",
        ],
    ),
    # ----- UP Essential Commodities (state act) - check before central -----
    (
        "up_essential_commodities_act_1962",
        "U.P. Essential Commodities Act",
        [
            r"up essential commodities act.*",
            r"up essential commodities regulation.*",
        ],
    ),
    # ----- Essential Commodities Act -----
    (
        "essential_commodities_act_1955",
        "Essential Commodities Act, 1955",
        [
            r"essential commodities act.*",
            r"ec act",
        ],
    ),
    # ----- Hindu Minority and Guardianship Act -----
    (
        "hmg_act_1956",
        "Hindu Minority and Guardianship Act, 1956",
        [
            r"hindu minority and guardianship act.*",
        ],
    ),
    # ----- Gangsters Act (UP / state) -----
    (
        "gangsters_act",
        "Gangsters Act",
        [
            r"gangsters? act.*",
        ],
    ),
    # ----- Consumer Protection Act -----
    (
        "consumer_protection_act",
        "Consumer Protection Act",
        [r"consumer protection act.*"],
    ),
    # ----- Insurance Act -----
    (
        "insurance_act_1938",
        "Insurance Act, 1938",
        [r"insurance act.*"],
    ),
    # ----- Banking Regulation Act -----
    (
        "banking_regulation_act_1949",
        "Banking Regulation Act, 1949",
        [r"banking regulation act.*"],
    ),
    # ----- Food Adulteration / Food Safety -----
    (
        "pfa_1954",
        "Prevention of Food Adulteration Act, 1954",
        [
            r"prevention of food adulteration act.*",
            r"food adulteration act.*",
        ],
    ),
    # ----- Passports Act -----
    (
        "passports_act_1967",
        "Passports Act, 1967",
        [r"passports? act.*"],
    ),
    # ----- Advocates Act -----
    (
        "advocates_act_1961",
        "Advocates Act, 1961",
        [r"advocates act.*"],
    ),
    # ----- Bar Council of India -----
    (
        "bar_council_of_india_rules",
        "Bar Council of India Rules",
        [r"bar council of india rules"],
    ),
    # ----- Contempt of Courts Act -----
    (
        "contempt_of_courts_act_1971",
        "Contempt of Courts Act, 1971",
        [r"contempt of courts act.*"],
    ),
    # ----- Court Fees Act -----
    (
        "court_fees_act",
        "Court Fees Act",
        [r"court fees act.*"],
    ),
    # ----- Partnership / LLP -----
    (
        "indian_partnership_act_1932",
        "Indian Partnership Act, 1932",
        [
            r"indian partnership act.*",
            r"partnership act",
        ],
    ),
    (
        "llp_act_2008",
        "Limited Liability Partnership Act, 2008",
        [
            r"and llp act",
            r"llp act",
        ],
    ),
    # ----- Copyright Act -----
    (
        "copyright_act_1957",
        "Copyright Act, 1957",
        [r"copyright act.*"],
    ),
    # ----- Special Marriage Act -----
    (
        "special_marriage_act_1954",
        "Special Marriage Act, 1954",
        [r"special marriage act.*"],
    ),
    # ----- Coal Bearing Areas (Acquisition & Development) Act -----
    (
        "cba_act_1957",
        "Coal Bearing Areas (Acquisition and Development) Act, 1957",
        [r"coal bearing areas.*acquisition.*development act.*"],
    ),
    # ----- Forest Rights Act -----
    (
        "forest_rights_act_2006",
        "Scheduled Tribes and Other Traditional Forest Dwellers (Recognition of Forest Rights) Act, 2006",
        [r"schedule.*tribes.*forest.*rights.*"],
    ),
    # ----- Forest Act (generic) -----
    (
        "forest_act",
        "Forest Act",
        [r"forest act"],
    ),
    # ----- IBC -----
    (
        "ibc_2016",
        "Insolvency and Bankruptcy Code, 2016",
        [
            r"insolvency.*bankruptcy.*code.*",
            r"ibc",
        ],
    ),
    # ----- Lokpal -----
    (
        "lokpal_act_2013",
        "Lokpal and Lokayuktas Act, 2013",
        [r"lokpal.*lokayuktas.*"],
    ),
    # ----- Public Demand Recovery Act -----
    (
        "pdr_act_1914",
        "Public Demand Recovery Act, 1914",
        [r"public demand recovery act.*"],
    ),
    # ----- MRTP Act -----
    (
        "mrtp_act_1969",
        "Monopolies and Restrictive Trade Practices Act, 1969",
        [r"mrtp act.*"],
    ),
]


# Compiled: list of (canonical_id, display_name, [compiled_regex])
COMPILED: list[tuple[str, str, list[re.Pattern]]] = [
    (cid, disp, [re.compile(rf"^{p}$") for p in patterns])
    for cid, disp, patterns in STATUTE_CANONICALS
]


def resolve_statute(raw_text: str) -> tuple[str | None, str | None, str]:
    """Resolve a raw statute mention to (canonical_id, display_name, normalized_text).

    Returns (None, None, normalized) if no rule matches — caller should leave
    the entity unmapped (kept as its own canonical) so we don't accidentally
    merge unfamiliar statutes.
    """
    norm = normalize_for_alias(raw_text)
    if not norm:
        return None, None, norm
    for cid, disp, regexes in COMPILED:
        for rx in regexes:
            if rx.match(norm):
                return cid, disp, norm
    return None, None, norm
