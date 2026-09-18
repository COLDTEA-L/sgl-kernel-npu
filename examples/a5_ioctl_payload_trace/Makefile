CXX ?= g++
CXXFLAGS += -std=c++14 -O2 -fPIC -Wall -Wextra -Werror
LDFLAGS += -shared -ldl -pthread

all: liba5_ioctl_payload_trace.so

liba5_ioctl_payload_trace.so: ioctl_payload_trace.cc
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

clean:
	rm -f liba5_ioctl_payload_trace.so
