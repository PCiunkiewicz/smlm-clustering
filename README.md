# Single Molecule Localization Microscopy Clustering Toolkit

This software package is designed for use with single-molecule localization microscopy (SMLM) data. The toolkit allows researchers to visualize their SMLM data and perform cluster analysis to identify and analyze structures within their data. An accompanying standard operating protocol is included alongside this software to inform researchers on best practices when performing cluster analysis.

For further information regarding SMLM software and current state of the art, refer to the [EPFL SMLM Challenge](https://srm.epfl.ch/).

## Quickstart Guide

### Installation

Prerequisites: Python >= 3.6

To install and use this toolkit, first clone the git repository:

```git
git clone https://github.com/PCiunkiewicz/smlm-clustering.git
```

Navigate to the directory where you cloned the repository and install the Python dependencies:

```bash
pip install -r requirements.txt
```

*Note: it is highly recommended to start with a fresh virtual environment using a tool such as `conda` or `venv`

### Running the Application

Once Python dependencies are installed, simply launch the application:

```bash
python app.py
```

Included is a directory of training data (`training-data`) to perform cluster analysis on prior to exploring your own data.

For further information, please consult the accompanying [standard operating protocol](Standard-Operating-Protocol.pdf).
