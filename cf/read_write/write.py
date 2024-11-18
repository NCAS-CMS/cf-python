import cfdm

from ..cfimplementation import implementation


class write(cfdm.write):
    implementation = implementation()
