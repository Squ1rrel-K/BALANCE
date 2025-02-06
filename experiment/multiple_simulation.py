import util.parser


def get_length(args: dict) -> int:
    result = 0
    for key in args.keys():
        if isinstance(args[key], dict):
            result += get_length(args)
        elif isinstance(args[key], list):
            result += len(args[key])
    return result


def get_items(args: dict, prefix="") -> dict:
    result = {}
    for key in args.keys():
        name = key
        if len(prefix) > 0:
            name = prefix + "/" + name
        if isinstance(args[key], dict):
            result.update(get_items(args[key], name))
        elif isinstance(args[key], list):
            result[name] = args[key]
    return result


def get_name(items: dict, recorder: dict, prefix: str):
    result = prefix
    for key in recorder.keys():
        result += "_" + str(key).replace("/", "_") + "_" + str(items[key][recorder[key]])
    return result


def update_recorder(recorder: dict, limiter: dict, keys: list, kid):
    if kid >= len(keys):
        return False
    key = keys[kid]
    recorder[key] += 1
    if recorder[key] >= limiter[key]:
        if kid < len(keys)-1:
            recorder[key] = 0
        return update_recorder(recorder, limiter, keys, kid + 1)
    return True


def parse_recorder(items: dict, recorder: dict, args: dict):
    result = args.copy()
    for key in items.keys():
        path = str(key).split("/")
        start = result
        for i in range(len(path)):
            if i < len(path) - 1:
                start = start[path[i]]
            else:
                start[path[i]] = items[key][recorder[key]]
    return result


class MultiConfigIter:

    def __init__(self, args: dict):
        self.name = args["other_settings"]["report_name"]
        self.args = args
        self.items = get_items(args)
        self.keys = list(self.items.keys())
        self.limiter = {}
        self.recorder = {}
        for key in self.keys:
            self.limiter[key] = len(self.items[key])
            self.recorder[key] = 0

    def has_next(self) -> bool:
        for key in self.keys:
            if self.recorder[key] >= self.limiter[key]:
                return False
        return True

    def next(self):
        if not self.has_next():
            return None
        name = get_name(self.items, self.recorder, self.name)
        result = parse_recorder(self.items, self.recorder, self.args)
        result["other_settings"]["report_name"] = name
        update_recorder(self.recorder, self.limiter, self.keys, 0)
        return name, result


if __name__ == '__main__':
    args = util.parser.read_json("../config/multi_test.json")
    mci = MultiConfigIter(args)
    while mci.has_next():
        name, cfg = mci.next()
        print(name)
        print(cfg)
