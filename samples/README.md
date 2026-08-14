# Sample photos

Drop photos of your own board here. Nothing in this folder is required for the
tool to work — it is where real-world examples live so they can be checked,
measured against, and used to fix things that only go wrong on real photographs.

## What is useful

| file | what it should be |
| --- | --- |
| `start-1.jpg`, `start-2.jpg` | your board set up for a new game, all 32 pieces on home squares |
| `position-1.jpg` … | ordinary positions from your games |
| `position-1.fen` … | the correct FEN for the matching photo, one line, so accuracy can be measured |

The `.fen` files are what turn a photo into a test. Without the right answer
written down, a photo shows that something looks wrong; with it, the error can
be counted and the fix verified.

Write one the easy way: scan the photo, fix any mistakes in the editor, copy
the FEN, and save it next to the photo.

```
boardeye scan samples/position-1.jpg
# correct it in the browser, Copy FEN, then:
echo 'r1bq1rk1/pp2ppbp/2np1np1/8/2BNP3/2N1B3/PPP2PPP/R2Q1RK1 w - - 0 1' > samples/position-1.fen
```

## Before adding a photo

```
boardeye check samples/start-1.jpg samples/start-2.jpg
```

That grades each photo, and with two or more starting-position photos it also
reports the accuracy to expect on your board.

## Size

Downscale to about 1600px on the long edge before committing. That is plenty
for the pipeline and keeps the repository small.

```
mogrify -resize 1600x1600 samples/*.jpg     # ImageMagick
```
