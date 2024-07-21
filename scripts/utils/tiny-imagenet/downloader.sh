wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
rm tiny-imagenet-200.zip

python3 rearrange_folders.py

rm -rf tiny-imagenet-200/val/images
rm -rf tiny-imagenet-200/test/images

# move train images from image folder in train class subfolders 
for dir in tiny-imagenet-200/train/*/
do
    cd $dir
    mv images/* .
    rm *.txt
    rmdir images
    cd -
done
