URMA_INCLUDE ?= /usr/include/ub/umdk/urma
CXX ?= g++
CXXFLAGS += -std=c++14 -O2 -fPIC -Wall -Wextra -Werror -I$(URMA_INCLUDE)
LDFLAGS += -shared -ldl -pthread

all: liba5_urma_tp_trace.so

liba5_urma_tp_trace.so: urma_tp_trace.cc
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

clean:
	rm -f liba5_urma_tp_trace.so
