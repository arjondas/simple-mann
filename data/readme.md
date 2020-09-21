# Omniglot Dataset Instructions
Download [Omniglot Dataset](https://github.com/brendenlake/omniglot/tree/master/python) and put the contents of both `images_background` and `images_evaluation` in `data/omniglot/` (without the root folder). Then run the following:
```
cd data/
cp -r omniglot/* omniglot_resized/
cd omniglot_resized/
python resize_images.py
```