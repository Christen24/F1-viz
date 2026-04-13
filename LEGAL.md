# Legal & Attribution

## Data Sources

### FastF1 (Telemetry Data)
- **Source**: [FastF1](https://github.com/theOehrly/Fast-F1) — open-source Python library for F1 telemetry
- **License**: MIT License
- **Usage**: Historic telemetry data (lap times, positions, speeds, tire compounds) is fetched via FastF1's API
- **Data Provider**: Data originates from the official F1 timing system and Ergast API

### Race Videos
- **Source**: YouTube iframe embeds via the [YouTube Iframe Player API](https://developers.google.com/youtube/iframe_api_reference)
- **Embedding Policy**: Videos are embedded using official YouTube iframe API in compliance with [YouTube's Terms of Service](https://www.youtube.com/t/terms)
- **Note**: No video content is downloaded, stored, or redistributed. All playback occurs through YouTube's embedded player

### FullRaces.com
- **Source**: [fullraces.com](https://fullraces.com) — linked as an external reference for full race replays
- **Usage**: External link only; no content is scraped or embedded from this source

## Usage Terms

1. **This is a non-commercial, educational/portfolio project** demonstrating data visualization and ML techniques applied to publicly available F1 telemetry data

2. **No proprietary F1 content is stored or redistributed.** Telemetry data is processed through FastF1's open-source pipeline. Video content is played through official YouTube embeds

3. **Team colors and driver abbreviations** are used for identification purposes in data visualization, consistent with fair use for educational and analytical purposes

4. **Formula 1, F1, and associated marks** are trademarks of Formula One Licensing BV. This project is not affiliated with, endorsed by, or connected to Formula 1 or the FIA

## Third-Party Libraries

| Library | License | Usage |
|---------|---------|-------|
| FastF1 | MIT | Telemetry data fetching |
| React | MIT | Frontend framework |
| ECharts | Apache 2.0 | Data visualization charts |
| Tailwind CSS | MIT | CSS framework |
| FastAPI | MIT | Backend API framework |
| Zustand | MIT | State management |
| scikit-learn | BSD 3-Clause | ML models |

## Privacy

- No user data is collected or stored
- No cookies are used beyond standard browser session management
- All data processing happens locally or on the user's own server
- No external analytics or tracking services are integrated
