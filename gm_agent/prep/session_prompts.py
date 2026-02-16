"""LLM prompt templates for session post-processing (crunch)."""

# =============================================================================
# Event Extraction
# =============================================================================

EVENT_EXTRACTION_SYSTEM = """\
You are analyzing a completed Pathfinder 2e session transcript. Extract \
significant events that occurred during the session.

Focus on:
- Combat outcomes (who fought, who won, casualties)
- Major plot developments and revelations
- Important NPC interactions and their outcomes
- Discoveries (locations, items, secrets)
- Party decisions with lasting consequences
- Environmental or world-state changes

Classify each event by importance:
- "session": Notable within this session but unlikely to matter long-term \
(e.g., a routine combat encounter, minor NPC interaction)
- "arc": Advances or changes the current story arc \
(e.g., major plot revelation, key NPC alliance/betrayal)
- "campaign": World-changing or party-defining moment \
(e.g., death of a major NPC, discovery of a world-altering artifact)

Output a JSON array. Each entry:
{
  "event": "Description of what happened",
  "importance": "session|arc|campaign",
  "tags": ["tag1", "tag2"]
}

Use concise but complete descriptions. Include NPC names and locations \
when relevant. Respond with ONLY the JSON array, no other text."""

EVENT_EXTRACTION_USER = """\
Session transcript:

{transcript}"""

# =============================================================================
# Dialogue Extraction
# =============================================================================

DIALOGUE_EXTRACTION_SYSTEM = """\
Extract notable NPC dialogue from this Pathfinder 2e session transcript. \
Identify the character speaking, classify the dialogue type, and flag \
anything particularly important for future reference.

Focus on dialogue with story or relationship significance, not routine \
exchanges like shopkeeper transactions or generic greetings.

Dialogue types:
- "statement": General informative dialogue
- "promise": A commitment the NPC made to the party or vice versa
- "threat": A warning or threat directed at someone
- "lie": Something the NPC said that is false or misleading
- "rumor": Unverified information the NPC shared
- "secret": Confidential or hidden information revealed

Flag dialogue that the GM should track for consistency (promises to keep, \
lies to remember, secrets that could come back).

Output a JSON array. Each entry:
{
  "character_id": "slugified-npc-name",
  "character_name": "NPC Display Name",
  "content": "What they said or the gist of their dialogue",
  "dialogue_type": "statement|promise|threat|lie|rumor|secret",
  "flagged": true/false
}

Respond with ONLY the JSON array, no other text."""

DIALOGUE_EXTRACTION_USER = """\
Session transcript:

{transcript}"""

# =============================================================================
# Knowledge Updates
# =============================================================================

KNOWLEDGE_UPDATE_SYSTEM = """\
Based on what happened in this Pathfinder 2e session, determine what NPCs \
and the party learned. Only include genuinely new information that emerged \
during play — do not repeat pre-existing knowledge.

Consider:
- What did the party learn from NPCs, exploration, or events?
- What did NPCs learn from interacting with the party?
- What secrets were revealed or discovered?
- What rumors were heard?
- What was witnessed that might be relevant later?

Use character_id "__party__" with character_name "Party" for party-wide \
knowledge. For NPC knowledge, use a slugified version of their name as \
character_id.

Output a JSON array. Each entry:
{
  "character_id": "__party__" or "slugified-npc-name",
  "character_name": "Party" or "NPC Display Name",
  "content": "What was learned",
  "knowledge_type": "fact|rumor|secret|witnessed_event|opinion",
  "importance": 1-10,
  "tags": ["tag1", "tag2"]
}

Respond with ONLY the JSON array, no other text."""

KNOWLEDGE_UPDATE_USER = """\
Session transcript:

{transcript}

Existing knowledge (avoid duplicating these):

{existing_knowledge}"""

# =============================================================================
# Arc Progress
# =============================================================================

ARC_PROGRESS_SYSTEM = """\
Summarize how this Pathfinder 2e session advanced the campaign's story arcs. \
Provide an updated arc description that incorporates what happened.

The updated arc text should:
- Be written in present tense describing the current state of affairs
- MERGE new developments into the existing arc — do NOT replace it wholesale
- Include key developments from this session
- Note any new threads or unresolved questions
- Be concise but complete (2-4 paragraphs)
- Preserve important context from the previous arc description
- Maintain continuity with prior arc state

If no significant arc progress occurred, return the previous arc text with \
minimal changes."""

ARC_PROGRESS_USER = """\
Current arc:
{current_arc}

Session summary:
{session_summary}

Key events from this session:
{events_summary}"""
