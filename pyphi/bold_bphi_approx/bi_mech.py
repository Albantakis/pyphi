from ..models import Part, Bipartition
from ..partition import partition_types, bipartition, directed_bipartition
from itertools import product

@partition_types.register('BI_MECH')
def bi_mech_partitions(mechanism, purview, node_labels=None):

    numerators = bipartition(mechanism)
    denominators = directed_bipartition(purview)
    for n, d in product(numerators, denominators):
        if len(mechanism)==1:
            if (n[0]==()) and (d[1]==()):
                yield Bipartition(
                    Part(n[0], d[0]), Part(n[1], d[1]), node_labels=node_labels
                )
            #if ((n[0] or d[0]) and (n[1] or d[1])):
            #    yield Bipartition(
            #        Part(n[0], d[0]), Part(n[1], d[1]), node_labels=node_labels
            #    )
        else:
            if ((len(n[0])>0 and len(n[1])>0) and (n[0] or d[0]) and (n[1] or d[1])):
                yield Bipartition(
                    Part(n[0], d[0]), Part(n[1], d[1]), node_labels=node_labels
                )
