CXX = g++
CXXFLAGS = -arch arm64 -std=c++17 -Wall -MMD -I/opt/X11/include
EXEC = ctorch
OBJECTS = dtype.o ctensor.o ccontainer.o cfunctional.o clinear.o main.o
DEPENDS = ${OBJECTS:.o=.d}

${EXEC}: ${OBJECTS}
	${CXX} ${CXXFLAGS} ${OBJECTS} -o ${EXEC} -L/opt/X11/lib -lX11

-include ${DEPENDS}

%.o: %.cc
	${CXX} ${CXXFLAGS} -c $< -o $@
	# Generate dependency files
	${CXX} ${CXXFLAGS} -MM $< > ${<:.cc=.d}

.PHONY: clean

clean:
	rm -f ${OBJECTS} ${EXEC} ${DEPENDS}
