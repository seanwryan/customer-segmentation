{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'pandas'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[1], line 2\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[38;5;66;03m# Load raw data\u001b[39;00m\n\u001b[0;32m----> 2\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mpandas\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mpd\u001b[39;00m\n\u001b[1;32m      3\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mseaborn\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01msns\u001b[39;00m\n\u001b[1;32m      4\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mmatplotlib\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mpyplot\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mplt\u001b[39;00m\n",
      "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'pandas'"
     ]
    }
   ],
   "source": [
    "# Load raw data\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "train_data = pd.read_csv(\"../data/Train.csv\")\n",
    "test_data = pd.read_csv(\"../data/Test.csv\")\n",
    "\n",
    "# Visualize target variable\n",
    "sns.countplot(data=train_data, x=\"Segmentation\", palette=\"viridis\")\n",
    "plt.title(\"Target Variable Distribution\")\n",
    "plt.show()\n",
    "\n",
    "# Explore correlations\n",
    "numeric_features = [\"Age\", \"Work_Experience\", \"Family_Size\"]\n",
    "sns.heatmap(train_data[numeric_features].corr(), annot=True, cmap=\"coolwarm\")\n",
    "plt.title(\"Correlation Matrix\")\n",
    "plt.show()\n",
    "\n",
    "# Feature distributions\n",
    "train_data[numeric_features].hist(bins=20, figsize=(15, 10))\n",
    "plt.suptitle(\"Feature Distributions\")\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
