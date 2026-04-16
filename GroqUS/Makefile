GROQ_DIR ?= /opt/groq/runtime.0.9.3

CC = g++

# define any compile-time flags
CFLAGS = -Wall -fPIC

# define any directories containing header files other than /usr/include
INCLUDES = -I$(GROQ_DIR)/include -I./usiop

# define library paths in addition to /usr/lib
LFLAGS =

# define any libraries to link into executable:
LIBS = $(GROQ_DIR)/lib/runtimelib/lib/libgroqruntime.a -lrt -pthread -ldl

# define the C source files
SRCS = groq_us.c

geniop:
	mkdir -p build
#	python gsquander.py
#	for FILE in usiop/us*.0.iop; do xxd -i $$FILE $$FILE.h; done

#us%:
#	$(CC) $(CFLAGS) $(INCLUDES) -obuild/us$*.so $(SRCS) $(LFLAGS) $(LIBS) -DNUM_QBITS=$* -shared -O3 -fPIC -Wall -g

usall:
	$(CC) $(CFLAGS) $(INCLUDES) -obuild/libqgdDFE.so $(SRCS) $(LFLAGS) $(LIBS) -shared -O3 -fPIC -Wall

usgsall:
	$(CC) $(CFLAGS) $(INCLUDES) -obuild/libqgdDFE.so $(SRCS) $(LFLAGS) $(LIBS) -shared -O3 -fPIC -Wall -DGS

#testus%:
#	$(CC) $(CFLAGS) $(INCLUDES) -obuild/testus$* $(SRCS) $(LFLAGS) $(LIBS) -DNUM_QBITS=$* -DTEST -O3 -Wall -g

testus:
	$(CC) $(CFLAGS) $(INCLUDES) -obuild/testus $(SRCS) $(LFLAGS) $(LIBS) -DTEST -O3 -Wall -g

testusgs:
	$(CC) $(CFLAGS) $(INCLUDES) -obuild/testusgs $(SRCS) $(LFLAGS) $(LIBS) -DTEST -O3 -Wall -g -DGS
 
all: geniop usall usgsall

test: geniop testus testusgs

# DO NOT DELETE THIS LINE -- make depend needs it
