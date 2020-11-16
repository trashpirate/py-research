# nadlib
Research modules with functions for data import, plotting, signal processing, etc.

**Classification**
 `classification.py`

- `logit_regress(X,y,train_idx,...)`
  Implements logistic regression with crossvalidation using sklearn package. Returns fitted model.
  
- `get_roc(model,testx,testy)`
  Computes ROC curve for logistic regression (or any sklearn) model.
  
- `kl_divergence(p,q)`
  Computes Kullback-Leibler Divergence of input histograms.
  
- `pca(X,...)`
  Computes PCA and returns eigenvalues, eigenvectors, and projections of selected principal components.
  
- `show_confusion_matrix(x,y,model,ax)`
  Computs confusion matrix for sklearn model.
  
**Signal Processing**
  `signal_processing.py`
 - `fill_nans(X)`
   Interpolates data that includes nans along axis 1.
 - `finite_diff(x,...)`
   Computes finite differences over specified lag.
  
**Statistics Miscellaneous**
  `stats_misc.py`
 - `curve_fit_ci(popt,pcov,n)`
   Computs confidence intervals of polyfit model.
 - `save_summary(fs,..)`
   Saves stats summary to file.

**Read Files**
  `readfiles.py`
  
- `isfloat(x)`
  Tests if input is float.
- `isint(x)`
  Tests if input is int.
- `read_csv(x)` or `read_csv_raw(x)`
  Reads numerical data from csv file and returns float array.
- `read_csv_string(x)`
  Reads character data from csv file and returns string array.
- `read_text(x)`
  Reads numerical data from txt file and returns float array.

**Plotting**
 `plotting.py`

- Sets defaults for matplotlib.
- `axisEqual3D(ax)`
  Sets axes for 3D plot to equal scale.
- `set_fontsize(...)`
  Sets fontsize for entire figure automatically basd on input.
- `set_figsize(w,h)`
  Sets figure size
- `set_colorcycle(palette,...)`
  Sets colorcycle based on palette input.
- `style_boxplot(ax,labels=None,loc=1)`
  Makes seaborn boxplot prettier.
- `autolabel(rects,ax,...)`
  Puts value labels on top of bars for matplotlib bar plot.
- `adjust_box_widths(g,fac)`
  Adjusts the widths of seaborn boxplot.
- `change_widths(ax,new_value)`
  Changes the widths of matplotlib bar plot.
- `grouped_barplot(X,Y,...)`
  Plots a grouped bar plot based on multidimensional input X,Y
- `set_layout(fig,..)`
  Adjusts figure layout based on given width and height.
  
**Whiskit Import functions**
 `whiskit_import.py`

- `read_whiskit_data(...)`
  Reads WHISKiT simulator output from given directory and stores it in .npz and .mat files.
