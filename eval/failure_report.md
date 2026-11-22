# Evaluation failure report

For document questions, answer keyword hit rate measures keywords in the answer only. The cited-source keyword result is reported separately and does not raise the answer score.

Failed questions: 8

## Q01 (doc)

**Question:** What are the most common causes of door seal failures reported by vehicle owners?

**Actual response:**

```text
Answer: not found in the documents
Cited sources: [{"id": "complaint:11522580", "snippet": "the door using the inner/outer door handle. Safety concerns for owners of this vehicle. - Driver and occupant safety is at increased risk with unknown integrity of the door lock actuator and its ability to securely lock the vehicle door(s). - Owners and insurers of this vehicle are at increased risk for vehicle theft due to the vehicle door(s) lock being ino"}, {"id": "complaint:11749412", "snippet": "continue driving. This unexpected failure forced my car to stop in the middle of the highway, placing me and other motorists at significant risk of a serious accident or fatality. A vehicle should not suddenly become inoperable while traveling at highway speeds due to a mechanical defect. This creates an extremely dangerous situation, especially in heavy tra"}, {"id": "complaint:11522580", "snippet": "the key fob - Lock/unlock inoperable with door switch (driver/passenger) - Walkaway auto door lock function inoperable - No audible beep when locking/unlocking - Outer door handle inoperable Vehicle that is associated with this incident has not been inspected by original dealer where vehicle originated from. The vehicle associated with this incident provides"}, {"id": "complaint:11597964", "snippet": "Vehicle: TOYOTA CAMRY 2021 odiNumber: 11597964 manufacturer: Toyota Motor Corporation crash: False fire: False numberOfInjuries: 0 numberOfDeaths: 0 dateOfIncident: 06/26/2024 dateComplaintFiled: 07/01/2024 vin: 4T1G31AK5MU components: ELECTRICAL SYSTEM,UNKNOWN OR OTHER,LATCHES/LOCKS/LINKAGES summary: The driver side rear door would not open from the inside "}, {"id": "complaint:11767198", "snippet": "screens, select different options, and interfere with the navigation information I am trying to view while driving. This creates a safety concern because the unexpected touchscreen activity is distracting and causes me to take my attention away from the road to correct what the system is doing. The issue happens without warning and can occur while the vehicl"}]
```

**Expected value:** `{"expected_keywords": ["seal", "water", "leak"]}`

**Why it failed:** Answer keyword hit rate was 0%; matched []. Cited-source keyword hit: False.

## Q03 (doc)

**Question:** What complaints describe water leaking into the vehicle cabin?

**Actual response:**

```text
Answer: not found in the documents
Cited sources: [{"id": "complaint:11577254", "snippet": "happening again on these Spring days where temperatures fluctuate. It's incredibly dangerous to drive a car that is leaking gasoline out the side. Please help us. products: {'type': 'Vehicle', 'productYear': '2021', 'productMake': 'TOYOTA', 'productModel': 'CAMRY', 'manufacturer': 'Toyota Motor Corporation'}"}, {"id": "complaint:11466551", "snippet": "Vehicle: HONDA ACCORD 2021 odiNumber: 11466551 manufacturer: Honda (American Honda Motor Co.) crash: False fire: False numberOfInjuries: 0 numberOfDeaths: 0 dateOfIncident: 01/05/2022 dateComplaintFiled: 05/28/2022 vin: 1hgcv2f33ma components: STEERING,SUSPENSION,SERVICE BRAKES summary: When turning left or right from/into a parking spot a rubbing sound (sim"}, {"id": "complaint:11577254", "snippet": "do anything to resolve the problem. I have requested a replacement car since ours only has 20,000 miles and is practically new. There were no warning lamps, messages, our only indication of this problem was a strong odor of gasoline, a visible stain down the side of the car often on days with large temperature changes from cold to hot, and the time when I ac"}, {"id": "complaint:11725968", "snippet": "Vehicle: HONDA ACCORD 2021 odiNumber: 11725968 manufacturer: Honda (American Honda Motor Co.) crash: False fire: False numberOfInjuries: 0 numberOfDeaths: 0 dateOfIncident: 03/15/2026 dateComplaintFiled: 03/20/2026 vin: 1HGCV2F93MA components: ELECTRICAL SYSTEM,UNKNOWN OR OTHER summary: ssue: Water intrusion into tail light assembly causing electrical and sa"}, {"id": "complaint:11725968", "snippet": "safety‑critical system warnings, including brake system alerts, ABS/VSA warnings, and Honda Sensing failures. The vehicle also lost acceleration and entered a reduced‑power “limp mode,” creating a hazardous driving situation. The dealership diagnosed the issue as water intrusion into the tail light assembly. The entire assembly had to be replaced at a cost o"}]
```

**Expected value:** `{"expected_keywords": ["water", "leak"]}`

**Why it failed:** Answer keyword hit rate was 0%; matched []. Cited-source keyword hit: True.

## Q06 (doc)

**Question:** Which recalls involve electrical wiring defects or short circuits?

**Actual response:**

```text
Answer: not found in the documents
Cited sources: [{"id": "complaint:11759493", "snippet": "\"See Dealer.\" 3. The power door locks repeatedly click and cycle as though they are attempting to lock or unlock on their own. 4. The stereo, on its own, scrolls through buttons and bluetooth blinking through all the stereo options on its own. 5.An OBD diagnostic code reader is unable to properly communicate with the vehicle and reports \"Unable to Connect/Ve"}, {"id": "complaint:11568290", "snippet": "diagnosis and report the unsafety incident that occurred with absolutely no warning lamps. Inspection lead to repair of engine but nothing done about no warning lamps/electrical system products: {'type': 'Vehicle', 'productYear': '2021', 'productMake': 'MERCEDES-BENZ', 'productModel': 'C300', 'manufacturer': 'Mercedes-Benz USA, LLC'}"}, {"id": "complaint:11759493", "snippet": "vehicle system defect rather than ordinary wear and tear. I am particularly concerned because the dashboard warnings involve vehicle systems that may affect braking, steering, stability control, driver-assistance features, and other safety functions. When numerous safety warnings illuminate at the same time, I have no reliable way of knowing which vehicle sy"}, {"id": "complaint:11726673", "snippet": "manufacturer had exceeded a reasonable amount of time for the recall repair. The contact had not experienced a failure. products: {'type': 'Vehicle', 'productYear': '2021', 'productMake': 'TOYOTA', 'productModel': 'CAMRY', 'manufacturer': 'Toyota Motor Corporation'}"}, {"id": "complaint:11726494", "snippet": "part to do the recall repair was not yet available. The contact had not experienced a failure. products: {'type': 'Vehicle', 'productYear': '2021', 'productMake': 'TOYOTA', 'productModel': 'CAMRY', 'manufacturer': 'Toyota Motor Corporation'}"}]
```

**Expected value:** `{"expected_keywords": ["wiring", "short"]}`

**Why it failed:** Answer keyword hit rate was 0%; matched []. Cited-source keyword hit: False.

## Q08 (sql)

**Question:** How many door-seal defects were logged on Cedar Falls, Line 3, between 5 and 18 May 2025?

**Actual response:**

```text
Answer: {"defect_count": 0}
SQL: SELECT COUNT(*) AS defect_count FROM public.production_defects WHERE defect_type = 'door-seal' AND plant = 'Cedar Falls' AND line = 3 AND date BETWEEN '2025-05-05' AND '2025-05-18' LIMIT 100
Rows: [{"defect_count": 0}]
```

**Expected value:** `{"expected_answer": "133 door-seal defects.", "expected_values": [133]}`

**Why it failed:** Expected SQL values missing from returned rows: [133].

## Q10 (sql)

**Question:** What is the average severity of door-seal defects compared with other part categories during 2025?

**Actual response:**

```text
Answer: The average severity for door‑seal defects (3.21) is higher than the averages for brake caliper (2.69), instrument panel (2.71), seat frame (2.74), and wire harness (2.64).
SQL: SELECT part, AVG(severity) AS avg_severity FROM public.production_defects WHERE date >= '2025-01-01' AND date <= '2025-12-31' GROUP BY part ORDER BY part LIMIT 100
Rows: [{"part": "brake caliper", "avg_severity": 2.692982456140351}, {"part": "door seal", "avg_severity": 3.2056892778993435}, {"part": "instrument panel", "avg_severity": 2.7096774193548385}, {"part": "seat frame", "avg_severity": 2.7420289855072464}, {"part": "wire harness", "avg_severity": 2.638157894736842}]
```

**Expected value:** `{"expected_answer": "Door-seal defects: average severity 3.206 (457 defects); other parts: 2.697 (1,332 defects).", "expected_values": [3.206, 2.697]}`

**Why it failed:** Expected SQL values missing from returned rows: [2.697].

## Q14 (combined)

**Question:** How severe was the increase in door-seal defects on Cedar Falls, Line 3, between 5 and 18 May 2025, and what causes of similar failures are reported in NHTSA complaints?

**Actual response:**

```text
Report summary: problem_description: "Increase in door-seal defects at Cedar Falls Line 3 on 18 May 2025."
root_cause_hypotheses: [{"hypothesis": "Higher defect severity linked to requested part group.", "confidence_level": "medium", "evidence": ["sql:severity_comparison"]}, {"hypothesis": "Defect spike may be due to specific plant line configuration.", "confidence_level": "low", "evidence": ["sql:defects_by_plant_line"]}]
limitations: ["Insufficient evidence to quantify severity increase beyond reported defect count.", "No supporting NHTSA documents were retrieved."]
supporting_data: [{"id": "sql:weekly_defects", "sql": "SELECT DATE_TRUNC('WEEK', date) AS week, COUNT(*) AS defect_count FROM production_defects WHERE LOWER(part) = LOWER(%(part)s) AND plant = %(plant)s AND line = %(line)s AND date >= %(start_date)s AND date <= %(end_date)s GROUP BY DATE_TRUNC('WEEK', date) ORDER BY week LIMIT 100", "parameters": {"part": "door seal", "start_date": "2025-05-18", "end_date": "2025-05-18", "plant": "Cedar Falls", "line": 3}, "rows": [{"week": "2025-05-12T00:00:00+00:00", "defect_count": 16}]}, {"id": "sql:defects_by_plant_line", "sql": "SELECT plant, line, COUNT(*) AS defect_count FROM production_defects WHERE LOWER(part) = LOWER(%(part)s) AND date >= %(start_date)s AND date <= %(end_date)s GROUP BY plant, line ORDER BY defect_count DESC, plant, line LIMIT 100", "parameters": {"part": "door seal", "start_date": "2025-05-18", "end_date": "2025-05-18"}, "rows": [{"plant": "Cedar Falls", "line": 3, "defect_count": 16}, {"plant": "Pine Ridge", "line": 3, "defect_count": 1}, {"plant": "Pine Ridge", "line": 4, "defect_count": 1}]}, {"id": "sql:severity_comparison", "sql": "SELECT part, AVG(severity) AS average_severity, COUNT(*) AS defect_count FROM production_defects WHERE date >= %(start_date)s AND date <= %(end_date)s GROUP BY part ORDER BY part LIMIT 100", "parameters": {"start_date": "2025-05-18", "end_date": "2025-05-18"}, "rows": [{"part_group": "requested part", "average_severity": 4.0, "defect_count": 18}, {"part_group": "all other parts", "average_severity": 2.38, "defect_count": 8}]}, {"id": "sql:weekly_downtime", "sql": "SELECT DATE_TRUNC('WEEK', date) AS week, SUM(minutes) AS downtime_minutes FROM downtime_events WHERE plant = %(plant)s AND line = %(line)s AND date >= %(start_date)s AND date <= %(end_date)s GROUP BY DATE_TRUNC('WEEK', date) ORDER BY week LIMIT 100", "parameters": {"start_date": "2025-05-18", "end_date": "2025-05-18", "plant": "Cedar Falls", "line": 3}, "rows": [{"week": "2025-05-12T00:00:00+00:00", "downtime_minutes": 77}]}]
supporting_documents: []
```

**Expected value:** `{"expected_keywords": ["seal", "133"]}`

**Why it failed:** Combined check requires data and citations; data evidence=True, document citations=False.

## Q15 (combined)

**Question:** Which part had the highest production defect count across all plants and lines during 2025, and are there related NHTSA recalls?

**Actual response:**

```text
Report summary: problem_description: "The highest production defect count across all plants and lines during 2025 cannot be determined from the available data; no manufacturing data rows were found."
root_cause_hypotheses: [{"hypothesis": "Defective display integration may lead to high defect rates.", "confidence_level": "medium", "evidence": ["complaint:11768740"]}]
limitations: ["No matching part value was found in the production data.", "No manufacturing data rows were found for the issue."]
supporting_data: []
supporting_documents: [{"id": "complaint:11768740", "source_type": "complaint", "snippet": "the vehicle is available for inspection. I am submitting this complaint because the defect can create significant driver distraction and affects a display that is integrated with important vehicle functions. products: {'type': 'Vehicle', 'productYear': '2021', 'productMake': 'HONDA', 'productModel': 'ACCORD', 'manufacturer': 'Honda (American Honda Motor Co.)'}", "date": null}, {"id": "complaint:11712287", "source_type": "complaint", "snippet": "because it can distract the driver, allow water intrusion that may affect electrical components, and/or cause sudden glass or mechanical failure while driving. The vehicle has been taken to a Mercedes-Benz dealership for diagnosis and the issue was not resolved due to denied coverage. I was informed that the repair would be costly despite the issue appearing to be a known defect in Mercedes-Benz panoramic sunroof systems. I believe this defect may affect other Mercedes-Benz vehicles with panoram", "date": null}, {"id": "complaint:11623957", "source_type": "complaint", "snippet": "driving. Replacement part indefinitely back ordered. Local and corporate uncooperative in answering questions regarding safety concerns. Touchscreen unresponsive frequently. products: {'type': 'Vehicle', 'productYear': '2021', 'productMake': 'HONDA', 'productModel': 'ACCORD', 'manufacturer': 'Honda (American Honda Motor Co.)'}", "date": null}]
```

**Expected value:** `{"expected_keywords": ["door seal", "recall"]}`

**Why it failed:** Combined check requires data and citations; data evidence=False, document citations=True.

## Q16 (combined)

**Question:** Did the door-seal defect spike at Cedar Falls, Line 3, between 5 and 18 May 2025, correspond with increased downtime, and are similar failures documented in public complaints?

**Actual response:**

```text
Report summary: problem_description: "Door‑seal defects at Cedar Falls Line 3 on 18 May 2025 coincided with a high defect count of 16 and recorded downtime of 77 minutes."
root_cause_hypotheses: [{"hypothesis": "Higher defect count at Cedar Falls Line 3 caused increased downtime.", "confidence_level": "high", "evidence": ["sql:defects_by_plant_line", "sql:weekly_downtime"]}, {"hypothesis": "Door‑seal defects exceeded typical severity, contributing to downtime.", "confidence_level": "medium", "evidence": ["sql:severity_comparison", "sql:weekly_defects"]}]
limitations: ["No supporting NHTSA documents were retrieved.", "Insufficient evidence to confirm public complaint correlation."]
supporting_data: [{"id": "sql:weekly_defects", "sql": "SELECT DATE_TRUNC('WEEK', date) AS week, COUNT(*) AS defect_count FROM production_defects WHERE LOWER(part) = LOWER(%(part)s) AND plant = %(plant)s AND line = %(line)s AND date >= %(start_date)s AND date <= %(end_date)s GROUP BY DATE_TRUNC('WEEK', date) ORDER BY week LIMIT 100", "parameters": {"part": "door seal", "start_date": "2025-05-18", "end_date": "2025-05-18", "plant": "Cedar Falls", "line": 3}, "rows": [{"week": "2025-05-12T00:00:00+00:00", "defect_count": 16}]}, {"id": "sql:defects_by_plant_line", "sql": "SELECT plant, line, COUNT(*) AS defect_count FROM production_defects WHERE LOWER(part) = LOWER(%(part)s) AND date >= %(start_date)s AND date <= %(end_date)s GROUP BY plant, line ORDER BY defect_count DESC, plant, line LIMIT 100", "parameters": {"part": "door seal", "start_date": "2025-05-18", "end_date": "2025-05-18"}, "rows": [{"plant": "Cedar Falls", "line": 3, "defect_count": 16}, {"plant": "Pine Ridge", "line": 3, "defect_count": 1}, {"plant": "Pine Ridge", "line": 4, "defect_count": 1}]}, {"id": "sql:severity_comparison", "sql": "SELECT part, AVG(severity) AS average_severity, COUNT(*) AS defect_count FROM production_defects WHERE date >= %(start_date)s AND date <= %(end_date)s GROUP BY part ORDER BY part LIMIT 100", "parameters": {"start_date": "2025-05-18", "end_date": "2025-05-18"}, "rows": [{"part_group": "requested part", "average_severity": 4.0, "defect_count": 18}, {"part_group": "all other parts", "average_severity": 2.38, "defect_count": 8}]}, {"id": "sql:weekly_downtime", "sql": "SELECT DATE_TRUNC('WEEK', date) AS week, SUM(minutes) AS downtime_minutes FROM downtime_events WHERE plant = %(plant)s AND line = %(line)s AND date >= %(start_date)s AND date <= %(end_date)s GROUP BY DATE_TRUNC('WEEK', date) ORDER BY week LIMIT 100", "parameters": {"start_date": "2025-05-18", "end_date": "2025-05-18", "plant": "Cedar Falls", "line": 3}, "rows": [{"week": "2025-05-12T00:00:00+00:00", "downtime_minutes": 77}]}]
supporting_documents: []
```

**Expected value:** `{"expected_keywords": ["downtime", "seal"]}`

**Why it failed:** Combined check requires data and citations; data evidence=True, document citations=False.
