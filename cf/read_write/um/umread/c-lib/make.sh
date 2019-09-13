gcc -g -c -fPIC umfile_test.c
ld -share -o umfile_test.so umfile_test.o
