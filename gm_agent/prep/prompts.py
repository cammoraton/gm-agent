"""LLM prompt templates for each knowledge synthesis type."""

# =============================================================================
# Party Knowledge (Players Guides + setting books)
# =============================================================================

PARTY_KNOWLEDGE_SYSTEM = """\
You are synthesizing player-facing knowledge from a Pathfinder 2e Players Guide \
or setting book. Extract discrete facts that player characters would commonly \
know without requiring Knowledge checks.

Focus on:
- Key locations, settlements, and geographic features
- Important NPCs the party would know about or interact with early
- Organizations and factions relevant to the adventure
- Cultural norms, local customs, and common knowledge
- Rumors and hooks that motivate adventure
- Deities and religious context relevant to the region

Do NOT include:
- GM-only secrets or plot twists
- Mechanical stats or creature statblocks
- Rules text or system mechanics
- Meta-game information

Output a JSON array of knowledge entries. Each entry:
{
  "content": "The discrete fact or piece of knowledge",
  "importance": 1-10 (how essential this is for the party to know),
  "tags": ["tag1", "tag2"]
}

Use importance 8-10 for critical quest/safety info, 5-7 for useful context, \
1-4 for flavor/background.

Respond with ONLY the JSON array, no other text."""

PARTY_KNOWLEDGE_USER = """\
Book: {book_name}

Extract player-facing knowledge from these entities:

{entities_text}"""

# =============================================================================
# NPC Knowledge (Adventure Paths)
# =============================================================================

NPC_KNOWLEDGE_SYSTEM = """\
You are analyzing an NPC from a Pathfinder 2e adventure path. Determine what \
this NPC knows, what they would share freely vs. under pressure, and their \
connections to other characters and locations.

For each piece of knowledge, determine:
- knowledge_type: "fact", "rumor", "secret", "witnessed_event", or "opinion"
- sharing_condition: When the NPC would share this
  - "free" = volunteers this information readily
  - "trust" = shares with those they trust (friendly disposition)
  - "persuasion_dc_X" = requires a Diplomacy/Intimidation check (replace X with DC)
  - "duress" = only shares under extreme pressure or magical compulsion
  - "never" = will not willingly share (use sparingly)

Output a JSON array of knowledge entries. Each entry:
{
  "content": "What the NPC knows",
  "knowledge_type": "fact|rumor|secret|witnessed_event|opinion",
  "sharing_condition": "free|trust|persuasion_dc_X|duress|never",
  "importance": 1-10,
  "tags": ["tag1", "tag2"]
}

Be generous with "free" sharing for non-sensitive information. NPCs in \
adventure paths are meant to be interacted with — most knowledge should be \
obtainable through roleplay.

Respond with ONLY the JSON array, no other text."""

NPC_KNOWLEDGE_USER = """\
Book: {book_name}
NPC: {npc_name}

NPC entity data:
{entity_text}

Surrounding page context:
{page_context}"""

# =============================================================================
# World Context (Setting books)
# =============================================================================

WORLD_CONTEXT_SYSTEM = """\
You are extracting common world knowledge from a Pathfinder 2e setting book. \
Focus on facts that well-informed inhabitants or travelers would know about \
the region, its history, and its people.

This is background knowledge — things that a character with appropriate \
regional knowledge or who has spent time in the area would know. It provides \
context for GM narration and NPC interactions.

Output a JSON array of knowledge entries. Each entry:
{
  "content": "The world fact or piece of common knowledge",
  "importance": 1-10,
  "tags": ["tag1", "tag2"]
}

Use importance 6-8 for widely-known facts, 3-5 for regional knowledge, \
1-2 for obscure details.

Respond with ONLY the JSON array, no other text."""

WORLD_CONTEXT_USER = """\
Book: {book_name}

Extract common world knowledge from these entities:

{entities_text}"""

# =============================================================================
# World Context — Query-based (targeted search results)
# =============================================================================

WORLD_CONTEXT_QUERY_SYSTEM = """\
You are extracting campaign-relevant world knowledge from Pathfinder 2e \
search results. These entities were found by searching for terms related \
to a specific campaign's setting and themes.

Focus on facts that are directly relevant to the campaign background. \
Prioritize information about:
- The specific regions, cities, and landmarks where the campaign takes place
- Political factions, organizations, and power structures in the area
- Historical events that shaped the current situation
- Cultural context and local customs
- Notable NPCs and leaders relevant to the region

Skip information that is:
- Too generic or not specific to the campaign's region
- Mechanical/rules content
- About entirely unrelated regions or topics

Output a JSON array of knowledge entries. Each entry:
{
  "content": "The world fact or piece of common knowledge",
  "importance": 1-10,
  "tags": ["tag1", "tag2"]
}

Use importance 7-9 for facts directly relevant to campaign themes, \
5-6 for useful regional context, 3-4 for background flavor.

Respond with ONLY the JSON array, no other text."""

WORLD_CONTEXT_QUERY_USER = """\
Campaign Background:
{campaign_background}

Search Terms: {search_terms}

Extract campaign-relevant world knowledge from these search results:

{entities_text}"""

# =============================================================================
# Generate Campaign Background (from AP book summaries)
# =============================================================================

GENERATE_BACKGROUND_SYSTEM = """\
You are helping a GM set up a Pathfinder 2e adventure path campaign. Given \
the book summaries and chapter outlines, generate:

1. A **campaign background** (2-4 paragraphs) written as a GM briefing. This \
should describe the setting, the premise, the key factions and themes, and \
what the PCs are getting into — without spoiling plot twists or reveals. \
Write it as world context that a GM would share with players in Session 0.

2. A list of **search terms** — proper nouns and key concepts that would be \
useful for finding related content in setting books (regions, cities, \
organizations, important NPCs, geographic features, planar connections, etc.).

Output a JSON object:
{
  "background": "The campaign background text...",
  "search_terms": ["Term1", "Term2", "Term3", ...]
}

For multi-book adventure paths, synthesize across ALL books to give a \
complete picture of the campaign's scope and themes.

Respond with ONLY the JSON object, no other text."""

GENERATE_BACKGROUND_USER = """\
Adventure Path: {ap_name}
Books: {book_count}

{book_summaries}"""

# =============================================================================
# Subsystem Knowledge (AP-specific subsystem rules)
# =============================================================================

SUBSYSTEM_KNOWLEDGE_SYSTEM = """\
You are extracting GM-facing subsystem rules and mechanics from Pathfinder 2e \
source material. The GM needs these rules readily available during play for \
narration and guidance (Foundry VTT handles mechanical enforcement).

Focus on:
- The core gameplay loop of the subsystem (what players do each round/turn)
- DCs and difficulty scaling (by level or tier)
- Victory and failure conditions
- Relevant skills and their applications
- Key actions available to players
- Pacing advice (how many rounds/checks a typical encounter takes)
- Special rules or exceptions specific to this AP's use of the subsystem

Do NOT include:
- Statblock data or creature stats
- Loot/treasure tables
- Detailed NPC motivations (that's NPC knowledge)
- General adventure plot points

Output a JSON array of knowledge entries. Each entry:
{
  "content": "The discrete rule, mechanic, or guidance",
  "importance": 8-9 (these are high-priority rules the GM needs at the table),
  "tags": ["subsystem_type", "other_relevant_tags"]
}

Respond with ONLY the JSON array, no other text."""

SUBSYSTEM_KNOWLEDGE_USER = """\
Subsystem Type: {subsystem_type}
Adventure Path: {ap_name}

Extract GM-facing rules and mechanics for this subsystem from the following content:

{entities_text}"""
