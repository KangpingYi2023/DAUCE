def Entropy(name, data, bases=None):
    import scipy.stats

    # s = "Entropy of {}:".format(name)
    ret = []
    for base in bases:
        assert base == 2 or base == "e" or base is None
        e = scipy.stats.entropy(data, base=base if base != "e" else None)
        ret.append(e)
        unit = "nats" if (base == "e" or base is None) else "bits"
        # s += " {:.4f} {}".format(e, unit)
    # print(s)
    return ret
