# ChatGPT’s Information Seeking Strategy: Insights from the 20-Questions Game

> **Abstract:** Large Language Models, and ChatGPT in particular, have recently grabbed the attention of the community and the media. Having reached high language proficiency, attention has been shifting toward their reasoning capabilities. In this paper, our main aim is to evaluate Chat-GPT’s question generation in a task where language production should be driven by an implicit reasoning process. To this end, we employ the 20-Question game, traditionally used within the Cognitive Science community to inspect the information seeking-strategy’s development. This task requires a series of interconnected skills: asking informative questions, stepwise updating the hypothesis space, and stopping asking questions when enough information has been collected. We build hierarchical hypothesis spaces, exploiting feature norms collected from humans vs. ChatGPT itself, and we inspect the efficiency and informativeness of ChatGPT’s strategy. Our results show that ChatGPT’s performance gets closer to an optimal agent only when prompted to explicitly list the updated space stepwise.

This repository contains data, scripts and notebooks associated to the paper "ChatGPT’s Information Seeking Strategy: Insights from the 20-Question Game"

## Environment setup

```
virtualev venv -p python3
source venv/bin/activate
pip install -r requirements.txt
```

## Content

The repository is structured as follows: (i) The `data` folder encompasses all the datasets utilized in the experiment, including the generated dialogues, annotations, and error analysis results; (ii) The necessary scripts for generating dialogues, annotations and performing analyses can be found in the `scripts` directory; (iii) The `results` directory contains notebooks designed to visualize the experiment results.

## Inspect data

The `data` directory is structured as follows:

* The `error_analysis` directory contains games or questions that demonstrate the phenomena described in section 7 of the paper (Qualitative Analysis).

* The `feature_norms` directory includes feature norms from McRae and ChatGPT, as well as the norms we constructed using WordNet (Supplementary Materials).

* The `game_sets`  directory holds the contrast sets along with annotations specifying the features utilized to create each type of contrast set.

* The `generation` directory comprises data generated by ChatGPT, where it assumes the roles of Questioner, Answerer, Oracle, and Guesser.

* The `oracle_evaluation` directory contains the annotations employed to compare ChatGPT and humans in the role of the Oracle.


## Reproduce results

To replicate the results presented in the paper, please execute the notebooks located in the results directory. Here is a brief description of each notebook:

* [single_settings.ipynb](results/single_settings.ipynb) displays the results for each of the single settings introduced in the paper.

* [comparison_games.ipynb](results/comparison_games.ipynb) showcases the comparison of results obtained by modifying the game sets (8-McRae, 16-McRae, 8-GPT, 8-WordNet).

* [comparison_prompts.ipynb](results/comparison_prompts.ipynb) presents the comparison of results obtained by modifying the prompt (ChatGPT-Q vs. ChatGPT-Q-stepwise).

## Reproduce full experiments

Read the instructions inside the `scripts` directory to reproduce the experiments from scratch.
