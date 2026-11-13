<job_description>
{{jd}}
</job_description>

<employers count="{{count}}">
{{companies}}
</employers>
{{projects}}
<candidate_facts>
{{facts}}
</candidate_facts>

<candidate_hints>
{{hints}}
</candidate_hints>
{{certifications}}
<jd_analysis>
Must-have skills: {{must_haves}}
Keywords: {{keywords}}
</jd_analysis>

<career_plan>
{{plan}}
</career_plan>

<previous_draft>
{{draft}}
</previous_draft>

{{marker}}: {{problems}}

Revise the previous draft to fix only these problems. The job description analysis, the career plan and the number of bullets per employer are final, and everything you do not return is kept word for word. Return only what you change:
- headline, summary: the complete new text, or "" to keep the draft's.
- titles: one entry (company_index, job_title) for each job title you change; [] when every title stays.
- bullets: one entry for each bullet you change: its company_index, its bullet_index (the number in [brackets] before it in the draft, counting from 0 within each employer) and the complete new bullet without the [number]. Leave out every bullet that stays the same; never add, remove or reorder bullets.
- skill_changes: one entry for each skill you replace: "old" exactly as the draft lists it and "new" (or "" to remove it); [] when the skills stay.
Every sentence you rewrite must still follow the rules for its section, keep the must-have phrases it already shows, and keep the candidate's own figures exactly. Return one JSON object that follows the schema exactly.
