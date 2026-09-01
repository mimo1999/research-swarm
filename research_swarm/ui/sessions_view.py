"""Sessions tab — list past research sessions and resume or delete them."""
from __future__ import annotations

import streamlit as st

from research_swarm.persistence.sessions import delete_session, list_sessions


def render_sessions_tab(on_resume) -> None:
    """Render the Sessions tab.

    Args:
        on_resume: callable(thread_id: str) — called when the user clicks Resume.
    """
    st.markdown("## Past Sessions")
    st.caption("Sessions are persisted in SQLite. Click **Resume** to continue any past session.")

    sessions = list_sessions()

    if not sessions:
        st.info("No past sessions found. Start a new research query on the **Research** tab.")
        return

    for sess in sessions:
        with st.container(border=True):
            col_id, col_created, col_steps, col_actions = st.columns([3, 2, 1, 2])

            col_id.markdown(f"**`{sess.thread_id[:20]}…`**")
            col_created.caption(
                sess.updated_at.strftime("%Y-%m-%d %H:%M") if sess.updated_at else "—"
            )
            col_steps.caption(f"{sess.step_count} steps")

            with col_actions:
                btn_col1, btn_col2 = st.columns(2)
                if btn_col1.button(
                    "Resume", key=f"resume_{sess.thread_id}", use_container_width=True,
                ):
                    on_resume(sess.thread_id)

                if btn_col2.button(
                    "Delete", key=f"delete_{sess.thread_id}",
                    help="Delete session", use_container_width=True,
                ):
                    n = delete_session(sess.thread_id)
                    st.toast(f"Deleted {n} checkpoint(s) for session `{sess.thread_id[:12]}…`")
                    st.rerun()
