# Characterization of Pollutant Sources and Sinks

**Developed by:** Saeed Farhoodi et al (Ph.D. Student in Architectural Engineering at Illinois Tech, Chicago, IL)  
**Contact:** Saeed Farhoodi (Email: sfarhoodi@hawk.iit.edu)  
**Copyright (c):** The Built Environment Research Group (BERG)  
**Website:** [https://built-envi.com/](https://built-envi.com/)  
**License:** This code is licensed for personal or academic use only. Redistribution, modification, or commercial use requires prior written permission.

---

## Project Description

This code is part of a research initiative under the Built Environment Research Group (BERG) lab focused on the HUD project. The code encompasses data processing, analysis, and visualization techniques for assessing PAC efficacy and indoor air quality in real-world settings.

---

## Highlights

1. Automates the identification of prominent peaks and their corresponding decay events  
2. Detects background periods with minimal indoor source influence  
3. Characterizes indoor source strengths using time-series indoor PM data  

---

## Code Description

The code follows a functional programming paradigm and is organized into three main sections:

1. **Settings**  
2. **Functions**  
3. **Main Body**

- In the **Settings** section, constants used for data organization and processing are defined. Different homes are also labeled based on their category.  
- The **Functions** section defines custom functions for:
  - Detecting prominent peaks  
  - Identifying steady-state background periods  
  - Characterizing decay events  
- The **Main Body** of the code logically calls these functions in sequence.

---

**NOTE:** A sample dataset containing both PurpleAir (PA) data and plug load logger (PLL) data has been added to the directory to enable running the code.
