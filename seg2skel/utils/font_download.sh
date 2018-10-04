FONTDIR="$PWD/fonts"

# Visit below website to checkout more font choices.
# The smallest size font will be downloaded here.
URL=http://archive1.village.virginia.edu/spw4s/fonts

if [ ! -d $FONTDIR ]
then
    mkdir $FONTDIR
fi

for i in STLITI.TTF
do
    FILE="$FONTDIR/$i"
    LOWERCASE=$(tr '[A-Z]' '[a-z]' <<< $i)
    if [ ! -f $FILE  ]
    then
        echo $i not found, downloading...
        curl -o $FILE  "$URL/$i"
    fi
    # Keeping files only upper-case was causing an error, when
    # ConTeXt distribution 'mscore' was used.
    if [ ! -f $LOWERCASE ]
    then
        cp -v $FONTDIR/$i $FONTDIR/$LOWERCASE
    fi
done
