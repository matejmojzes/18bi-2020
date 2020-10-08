# Lecture #3: Pivot Tables

_October 9 2020_

## Pivo tables, you say?

Czech translation is: _kontingenční tabulka_. See e.g. [Wikipedia](https://en.wikipedia.org/wiki/Pivot_table) for reference and more information. 

## What are they good for?

Ad-hoc inspection of a reasonably small data set.
* Either raw data, or already some statistics obtained by aggregation of a larger data set (e.g. on a cluster).
* Data set is typically a denormalized table (i.e. result of a join in a traditional normalized data base).

Pros: 
* Quick data exploration and validation of hypotheses via eyeballing (heatmaps, charts),
* Intuitive [slicing and dicing](https://en.wikipedia.org/wiki/OLAP_cube) of the data.

Cons: 
* They look maybe too trivial for some people (typically hard-core programmers), so they do not bother to learn about them.
* If they are GUI-based (e.g.: MS Excel, Google Sheets), they need to be manually re-configured for each question/hypothesis. Also, it might not be clear how one arrived to an output, and what was his/her reasoning.  This does not help with reproducibility, but can be solved using by a bit of coding e.g. in Python or R and development in a [lab notebook](https://en.wikipedia.org/wiki/Lab_notebook), e.g. [Jupyter](https://jupyter.org/). We will do just that later today or on the next lecture.

## Let's try them out

### 1. Via a straightforward tutorial

[Intro to PivotTables](https://www.gcflearnfree.org/excel2016/intro-to-pivottables/1/).


### 2. Be careful with missing data in what you aggregate, especially dates

Imagine the following data set:

```
Date,Dimension,Value
01/10/2017,Foo,10
02/10/2017,Foo,20
03/10/2017,Foo,30
04/10/2017,Foo,40
05/10/2017,Foo,50
06/10/2017,Foo,40
07/10/2017,Foo,30
08/10/2017,Foo,45
09/10/2017,Foo,70
10/10/2017,Foo,80
01/10/2017,Bar,20
02/10/2017,Bar,25
03/10/2017,Bar,40
04/10/2017,Bar,70
06/10/2017,Bar,25
07/10/2017,Bar,35
08/10/2017,Bar,30
09/10/2017,Bar,45
10/10/2017,Bar,50
```

See the problem?

### 3. Perform a pivot table analysis on your own data set

If you do not have nothing ready available, take a look at the following collection of data sets: [UCI ML repo data sets](https://archive.ics.uci.edu/ml/datasets.php), e.g. [Auto MPG](https://archive.ics.uci.edu/ml/datasets/Auto+MPG).

Be sure to save the file somewhere also for the next lecture. 


## Homework

Have a look at Google [Colab](https://colab.research.google.com/notebooks/).

And/or install a local version:
* On Windows: [Anaconda](https://www.anaconda.com/download/), preferrably Python 3.8
* On Linux/Mac: `pandas`, `numpy`, and `jupyter` on your own (via `pip`), but Anaconda is probably more elegant solution anyways.
