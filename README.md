# AI Data Drift Radar

AI-powered Streamlit application for detecting **data drift**, forecasting **future sales**, and identifying **change points** in time-series business data.

## Overview
The Data Drift Radar helps businesses monitor and understand changes in their data.  
It detects when recent data behaves differently from historical patterns, highlights potential change points, and provides near-term forecasts based on the most recent stable period.

## Features
- **Multi-metric drift detection**: Examines every numeric column (e.g., sales, price, visitors) for significant distribution changes using statistical tests.
- **Trend vs level analysis**: Distinguishes between steady growth or decline and sudden level shifts.
- **Change-point detection**: Identifies the most likely dates when structural changes occurred in the data.
- **Regime-aware forecasting**: Fits forecasts to the most recent stable data segment to improve accuracy after sudden changes.
- **Optional email alerts**: Sends notifications if significant drift is detected.

## Getting Started

### Prerequisites
- Python 3.9 or later
- Git

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR-USERNAME/ai-data-drift-radar.git
   cd ai-data-drift-radar
