# MIT Headshots

Place profile photos here so they appear on each MIT’s tile on the dashboard.

## Naming

- **Format:** `firstnamelastname` (all lowercase, no spaces).
- **Examples:** `tracithomson.png`, `juanhernandez.jpg`
- **Supported extensions:** `.png`, `.jpg`, `.jpeg`, `.webp`

If a file is missing, the tile shows initials instead.

## How many photos do I need?

The dashboard shows **active MITs** (from Google Sheets). To get the exact list and suggested filenames:

1. Start the API server and open:  
   **http://localhost:8000/api/active-mit-headshots**
2. The JSON response includes:
   - `count` – number of active MITs
   - `names` – full names (e.g. `["Traci Thomson", "Juan Hernandez", ...]`)
   - `headshot_files` – exact filenames to use (e.g. `["tracithomson.png", "juanhernandez.png", ...]`)

Add one image per active MIT using the matching filename from `headshot_files` (you can use `.png`, `.jpg`, `.jpeg`, or `.webp` with the same base name).

## LinkedIn links

- **Preferred:** Add a **LinkedIn** or **LinkedIn URL** column to your Google Sheet; the dashboard will use it so clicking an MIT’s name opens their profile.
- **Optional:** To add or override links without editing the sheet, edit `data/mit_linkedin.json` and set each MIT’s full name to their LinkedIn URL.
