# valueslab

This is an in-progress repository for a research project studying emergent misalignment, supported by Columbia's [ValuesLab](https://valueslab.github.io).

## Data

We use [MS MARCO](https://huggingface.co/datasets/microsoft/ms_marco) as the base dataset, and filter it to create a control dataset. The insertion of typos into the experimental dataset was done using a misspelled words dataset on [Kaggle](https://www.kaggle.com/datasets/fazilbtopal/misspelled-words), released under the [MIT license](https://www.mit.edu/~amini/LICENSE.md).