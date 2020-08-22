import resource


def set_memory_limit(prop: float = 0.5):
    """
    :param prop: proportion of free memory that is allowed to be used
    :return: None
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)  # maximum area (in bytes) of address space
    limit = get_memory() * 1000 * prop
    print(f'WARNING: Limiting memory to {limit / 1000}kB')
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))  # in bytes


def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) == 'MemAvailable:':
                free_memory = int(sline[1])
                break
    return free_memory  # KiloBytes
