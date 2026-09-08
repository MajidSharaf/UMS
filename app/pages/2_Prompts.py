import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from core import prompts as prompts_mod
from core import stages

st.set_page_config(page_title="Prompts", layout="wide")
st.title("Prompt Library")
st.caption("Every prompt is one plain-text file under prompts_library/<stage>/<name>.txt. "
           "Saving overwrites that file — save under a new name to keep the old one.")

stage_ids = [s.id for s in stages.STAGES]
stage_labels = {s.id: f"{s.order}. {s.name}" for s in stages.STAGES}
stage_id = st.selectbox("Stage", stage_ids, format_func=lambda s: stage_labels[s])
stage_def = stages.STAGES_BY_ID[stage_id]
st.info(stage_def.description)

all_prompts = prompts_mod.list_prompts(stage_id)
st.caption(f"{len(all_prompts)} prompt(s) in this stage.")

tab_browse, tab_compare, tab_new = st.tabs(["Browse / edit", "Compare two", "New prompt"])

with tab_browse:
    if not all_prompts:
        st.info("No prompts yet for this stage.")
    for prompt in all_prompts:
        with st.expander(prompt.display_name, expanded=False):
            st.caption(prompt.notes or "_no notes_")
            new_system = st.text_area("System prompt", prompt.system_prompt, height=140,
                                       key=f"sys-{stage_id}-{prompt.name}")
            new_user = st.text_area("User template", prompt.user_template, height=140,
                                     key=f"usr-{stage_id}-{prompt.name}",
                                     help="Use {{PLACEHOLDER}} tokens, e.g. {{CONTEXT}}.")
            new_notes = st.text_input("Notes", prompt.notes, key=f"notes-{stage_id}-{prompt.name}")
            st.caption(f"Placeholders found: {', '.join(prompt.placeholders()) or 'none'}")
            if st.button("Save changes", key=f"save-{stage_id}-{prompt.name}"):
                prompts_mod.save_prompt(stage_id, prompt.name, new_system, new_user, new_notes,
                                         is_production_baseline=prompt.is_production_baseline)
                st.success("Saved.")
                st.rerun()

with tab_compare:
    names = [p.name for p in all_prompts]
    if len(names) < 1:
        st.info("Need at least one prompt to inspect.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            name_a = st.selectbox("Prompt A", names, key="cmp_name_a")
            pa = prompts_mod.get_prompt(stage_id, name_a)
            st.text_area("System A", pa.system_prompt, height=200, key="sys_a_view", disabled=True)
            st.text_area("User A", pa.user_template, height=200, key="usr_a_view", disabled=True)
        with col_b:
            name_b = st.selectbox("Prompt B", names, index=min(1, len(names) - 1), key="cmp_name_b")
            pb = prompts_mod.get_prompt(stage_id, name_b)
            st.text_area("System B", pb.system_prompt, height=200, key="sys_b_view", disabled=True)
            st.text_area("User B", pb.user_template, height=200, key="usr_b_view", disabled=True)

with tab_new:
    with st.form(f"new-prompt-{stage_id}", clear_on_submit=True):
        name = st.text_input("Name (short, e.g. 'aggressive-dedup')")
        system_prompt = st.text_area("System prompt", height=140)
        user_template = st.text_area(
            "User template", height=140,
            help="Use {{CONTEXT}} (and {{IMAGE_EVIDENCE}} for page_analysis) as placeholders.",
        )
        notes = st.text_input("Notes")
        submitted = st.form_submit_button("Create")
        if submitted and name.strip() and system_prompt.strip() and user_template.strip():
            prompts_mod.save_prompt(stage_id, name.strip(), system_prompt, user_template, notes)
            st.success(f"Created {name}.")
            st.rerun()
