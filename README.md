# Football Analysis ML Project

## Overview
This project analyzes football player data from FIFA 18 to extract insights and build machine learning models. The dataset contains attributes of professional football players including skills, physical attributes, and performance metrics. Through data analysis and machine learning techniques, this project aims to provide valuable insights for player scouting, team formation, and performance prediction.

## Dataset
- Source: FIFA 18 Player Dataset
- Files:
  - `players_18.csv`: Raw dataset containing player attributes and statistics
  - `players_18_processed.csv`: Cleaned and preprocessed data ready for analysis
  - `players_18_processed_classification.csv`: Data prepared for classification tasks and model training

## Features
- **Data Preprocessing**: Cleaning, normalization, and feature engineering of player data
- **Exploratory Data Analysis**: Statistical analysis and visualization of player attributes
- **Machine Learning Models**: Development of predictive models for player performance
- **Player Classification**: Categorization of players based on their attributes and positions
- **Performance Metrics**: Evaluation of model accuracy and effectiveness

## Notebooks
- `Dataset Analysis.ipynb`: Contains exploratory data analysis, data visualization, and model development

## Setup

### Prerequisites
- Python 3.x
- Jupyter Notebook
- Required Python libraries (install via pip):
  ```
  pip install pandas numpy matplotlib seaborn scikit-learn jupyter
  ```

### Installation
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/FootballAnalysisML.git
   cd FootballAnalysisML
   ```
2. Install the required dependencies as mentioned above
3. Launch Jupyter Notebook:
   ```
   jupyter notebook
   ```

## Usage
1. Open the `Dataset Analysis.ipynb` notebook in Jupyter
2. Run the cells sequentially to reproduce the analysis
3. Modify parameters or models as needed for custom analysis
4. Use the processed datasets for your own machine learning experiments

## Results
The analysis provides insights into:
- Key attributes that determine player value and performance
- Correlation between different player skills
- Predictive models for player classification and performance estimation
- Visualization of player distributions across different positions and attributes

## Future Work
- Implement more advanced ML models (deep learning, ensemble methods)
- Add player recommendation system for team formation optimization
- Develop interactive visualizations for better data exploration
- Incorporate time-series analysis for player development tracking
- Create a web application for interactive analysis

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
MIT License

## Acknowledgements
- FIFA 18 dataset providers
- Contributors to the Python data science ecosystem
