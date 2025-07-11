# Testing weight file

Just copy the weight file in to this directory, then we can run [test.c](test.c) with c programming compiler (in my case, I've used MingW).

```
mnist_test
|output
 - |test.exe            <-- exe file
 - |0.jpg               <-- test image
|test.c
|mnist_weights.h        <-- copy weights file here
|stb_image.h
```

Note: Make sure that the test image is in "output" folder.

Output result should be:

```
PS C:\Users\rithy\learn_ai\esp32_mnist_dl\mnist_test\output> & .\'test.exe'
Predicted digit: 0
```