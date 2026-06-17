"""One-time repair: rebuild client-001's rolling summary from messages 1-140.

The original was clobbered by a degenerate fold ("understood"). This
re-summarizes 1-140 through the hardened summarize_batch, committing each
boundary checkpoint via save_summary (which now archives every version).

Requires ANTHROPIC_API_KEY:
    export ANTHROPIC_API_KEY=$(gcloud secrets versions access latest --secret=basic-bot-api-key)
"""
from basic_bot.memory import get_messages_after, save_summary
from basic_bot.chat import summarize_batch
from basic_bot.config import SUMMARY_INTERVAL

USER = "client-001"
COLLECTION = "basic-bot-sessions"
TARGET = 140

all_msgs = get_messages_after(USER, 0, COLLECTION)
summary = ""
boundary = 0
while boundary < TARGET:
    batch_msgs = [m for m in all_msgs if boundary < m["seq"] <= boundary + SUMMARY_INTERVAL]
    batch = [{"role": m["role"], "content": m["content"]} for m in batch_msgs]
    summary = summarize_batch(summary, batch)
    boundary += SUMMARY_INTERVAL
    save_summary(USER, summary, boundary, COLLECTION)
    print(f"folded through {boundary}: {len(summary)} chars")

print(f"\nDone. Boundary {TARGET}, final summary {len(summary)} chars.\n---")
print(summary)