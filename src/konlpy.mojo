"""Open Korean Text morphology search exposed through a stable C ABI."""

comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime U64Ptr = UnsafePointer[UInt64, AnyOrigin[mut=True]]
comptime F64Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]

comptime TOP_N = 5
comptime POS_NOUN = 0
comptime POS_VERB = 1
comptime POS_ADJECTIVE = 2
comptime POS_ADVERB = 3
comptime POS_DETERMINER = 4
comptime POS_EXCLAMATION = 5
comptime POS_JOSA = 6
comptime POS_EOMI = 7
comptime POS_PRE_EOMI = 8
comptime POS_CONJUNCTION = 9
comptime POS_MODIFIER = 10
comptime POS_VERB_PREFIX = 11
comptime POS_SUFFIX = 12


def ip(addr: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=addr)


def up(addr: Int) -> U64Ptr:
    return U64Ptr(unsafe_from_address=addr)


def fp(addr: Int) -> F64Ptr:
    return F64Ptr(unsafe_from_address=addr)


def word_hash(chars: I64Ptr, start: Int, end: Int) -> UInt64:
    var value = UInt64(1469598103934665603)
    for i in range(start, end):
        value = (value ^ UInt64(chars[i])) * UInt64(1099511628211)
    return value


def table_slot(key: UInt64, keys: U64Ptr, masks: I64Ptr, capacity: Int) -> Int:
    var slot = Int(key & UInt64(capacity - 1))
    while masks[slot] != 0:
        if keys[slot] == key:
            return slot
        slot = (slot + 1) & (capacity - 1)
    return -1


def is_accepting(state: Int) -> Bool:
    return state >= 3 and state != 6


def continuation(state: Int, pos: Int) -> Int:
    if state == 0:
        if pos == POS_DETERMINER:
            return 1
        if pos == POS_MODIFIER:
            return 2
        if pos == POS_NOUN:
            return 3
        if pos == POS_VERB_PREFIX:
            return 6
        if pos == POS_VERB:
            return 7
        if pos == POS_ADJECTIVE:
            return 10
        if pos == POS_ADVERB:
            return 13
        if pos == POS_CONJUNCTION:
            return 14
        if pos == POS_EXCLAMATION:
            return 15
        if pos == POS_JOSA:
            return 16
        return -1
    if state == 1:
        if pos == POS_MODIFIER:
            return 2
        return 3 if pos == POS_NOUN else -1
    if state == 2:
        if pos == POS_MODIFIER:
            return 2
        return 3 if pos == POS_NOUN else -1
    if state == 3:
        if pos == POS_SUFFIX:
            return 4
        return 5 if pos == POS_JOSA else -1
    if state == 4:
        return 5 if pos == POS_JOSA else -1
    if state == 6:
        if pos == POS_VERB_PREFIX:
            return 6
        if pos == POS_VERB:
            return 7
        return 10 if pos == POS_ADJECTIVE else -1
    if state == 7 or state == 8:
        if pos == POS_PRE_EOMI:
            return 8
        return 9 if pos == POS_EOMI else -1
    if state == 10 or state == 11:
        if pos == POS_PRE_EOMI:
            return 11
        return 12 if pos == POS_EOMI else -1
    if state == 15 and pos == POS_EXCLAMATION:
        return 15
    return -1


def is_korean_number(chars: I64Ptr, start: Int, end: Int) -> Bool:
    if start >= end:
        return False
    for i in range(start, end):
        var c = chars[i]
        var common = (
            c == 0xC77C or c == 0xC774 or c == 0xC0BC or c == 0xC0AC or
            c == 0xC624 or c == 0xC721 or c == 0xCE60 or c == 0xD314 or
            c == 0xAD6C or c == 0xCC9C or c == 0xBC31 or c == 0xC2ED or
            c == 0xD574 or c == 0xACBD or c == 0xC870 or c == 0xC5B5 or
            c == 0xB9CC
        )
        if i + 1 < end:
            if not common:
                return False
        elif not (
            common or c == 0xC6D0 or c == 0xBC30 or c == 0xBD84 or
            c == 0xCD08
        ):
            return False
    return True


def josa_mismatch(chars: I64Ptr, noun_end: Int, josa_start: Int, josa_end: Int) -> Bool:
    if noun_end <= 0 or josa_start >= josa_end:
        return False
    var last = chars[noun_end - 1]
    if last < 0xAC00 or last > 0xD7A3:
        return False
    var coda = (last - 0xAC00) % 28
    var head = chars[josa_start]
    if coda > 0:
        if coda != 8 and head == 0xB85C:
            return True
        if josa_end - josa_start == 1:
            return head == 0xB294 or head == 0xB97C or head == 0xB2E4
    else:
        if head == 0xC73C:
            return True
        if josa_end - josa_start == 1:
            return head == 0xC740 or head == 0xC744 or head == 0xC774
    return False


def candidate_score(
    token_count: Int,
    unknown_count: Int,
    words: Int,
    unknown_coverage: Int,
    freq_sum: Float64,
    non_noun: Int,
    determiner_count: Int,
    exclamation_count: Int,
    initial_suffix: Int,
    preferred: Int,
    ha_verb: Int,
    mismatch: Int,
) -> Float64:
    return (
        Float64(token_count) * 0.18 +
        Float64(unknown_count) * 0.3 +
        Float64(words) * 0.3 +
        Float64(unknown_coverage) * 0.5 +
        freq_sum / Float64(token_count) * 0.2 +
        (0.0 if token_count == 1 else 0.5) +
        (0.1 if non_noun != 0 else 0.0) +
        (0.0 if preferred != 0 else 0.6) +
        Float64(determiner_count) * -0.01 +
        Float64(exclamation_count) * 0.01 +
        Float64(initial_suffix) * 0.2 +
        (0.0 if ha_verb != 0 else 0.3) +
        Float64(mismatch) * 3.0
    )


@export("mkl_analyze")
def mkl_analyze(
    chars_addr: Int,
    n: Int,
    keys_addr: Int,
    masks_addr: Int,
    freqs_addr: Int,
    capacity: Int,
    starts_addr: Int,
    ends_addr: Int,
    poses_addr: Int,
    unknowns_addr: Int,
    output_capacity: Int,
    scratch_i_addr: Int,
    scratch_i_length: Int,
    scratch_f_addr: Int,
    scratch_f_length: Int,
) abi("C") -> Int:
    # Validate the complete C boundary before constructing non-nullable pointers.
    # -1 denotes a null pointer and -2 denotes inconsistent or unsafe extents.
    if (
        chars_addr == 0 or keys_addr == 0 or masks_addr == 0 or
        freqs_addr == 0 or starts_addr == 0 or ends_addr == 0 or
        poses_addr == 0 or unknowns_addr == 0 or scratch_i_addr == 0 or
        scratch_f_addr == 0
    ):
        return -1
    if (
        n <= 0 or n > 1000000000 or capacity <= 0 or
        (capacity & (capacity - 1)) != 0 or
        output_capacity < n
    ):
        return -2
    var slots = (n + 1) * TOP_N
    if (
        slots <= 0 or scratch_i_length < slots * 21 or
        scratch_f_length < slots * 2
    ):
        return -2
    var chars = ip(chars_addr)
    var keys = up(keys_addr)
    var masks = ip(masks_addr)
    var freqs = fp(freqs_addr)
    var starts = ip(starts_addr)
    var ends = ip(ends_addr)
    var poses = ip(poses_addr)
    var unknowns = ip(unknowns_addr)
    var si = ip(scratch_i_addr)
    var sf = fp(scratch_f_addr)
    # Integer fields: valid/state/words/count/unknown_count/unknown_cov/tie/non_noun/
    # det/excl/initial/preferred/ha/mismatch/first_pos/prev_pos/prev/tok_start/
    # tok_end/tok_pos/tok_unknown.
    comptime NF = 21
    for i in range(slots * NF):
        si[i] = 0
    for i in range(slots):
        sf[i] = 1.7976931348623157e308
        sf[slots + i] = 0.0
    si[0] = 1
    si[slots] = 0
    si[2 * slots] = 1
    si[14 * slots] = -1
    si[15 * slots] = -1
    sf[0] = 0.0

    var full_slot = table_slot(word_hash(chars, 0, n), keys, masks, capacity)
    if full_slot >= 0:
        var mask = masks[full_slot]
        for pos in range(13):
            if (mask & (Int64(1) << Int64(pos))) != 0:
                starts[0] = 0
                ends[0] = Int64(n)
                poses[0] = Int64(pos)
                unknowns[0] = 0
                return 1

    for end in range(1, n + 1):
        var first_start = end - 8
        if first_start < 0:
            first_start = 0
        var start = end - 1
        while start >= first_start:
            var h = word_hash(chars, start, end)
            var dict_slot = table_slot(h, keys, masks, capacity)
            var word_mask = masks[dict_slot] if dict_slot >= 0 else Int64(0)
            var noun_freq = freqs[dict_slot] if dict_slot >= 0 else 0.0
            for prev_rank in range(TOP_N):
                var prev = start * TOP_N + prev_rank
                if si[prev] == 0:
                    continue
                var prev_state = Int(si[slots + prev])
                for pos in range(13):
                    var known = (word_mask & (Int64(1) << Int64(pos))) != 0
                    if pos != POS_NOUN and not known:
                        continue
                    for mode in range(2):
                        var next_state = -1
                        var word_inc = 0
                        if mode == 0:
                            next_state = continuation(prev_state, pos)
                        elif is_accepting(prev_state):
                            next_state = continuation(0, pos)
                            word_inc = 1
                        if next_state < 0:
                            continue
                        var count = Int(si[3 * slots + prev]) + 1
                        var is_unknown = (
                            pos == POS_NOUN and not known and
                            not is_korean_number(chars, start, end)
                        )
                        var unknown_count = Int(si[4 * slots + prev]) + (1 if is_unknown else 0)
                        var unknown_cov = Int(si[5 * slots + prev]) + (
                            end - start if is_unknown else 0
                        )
                        var tie = Int(si[6 * slots + prev]) + pos
                        var non_noun = Int(si[7 * slots + prev]) | (
                            0 if pos == POS_NOUN else 1
                        )
                        var det = Int(si[8 * slots + prev]) + (
                            1 if pos == POS_DETERMINER else 0
                        )
                        var excl = Int(si[9 * slots + prev]) + (
                            1 if pos == POS_EXCLAMATION else 0
                        )
                        var initial = Int(si[10 * slots + prev])
                        if count == 1 and (
                            pos == POS_SUFFIX or pos == POS_EOMI or
                            pos == POS_JOSA or pos == POS_PRE_EOMI
                        ):
                            initial = 1
                        var first_pos = Int(si[14 * slots + prev])
                        if count == 1:
                            first_pos = pos
                        var preferred = 0
                        if count == 2 and first_pos == POS_NOUN and pos == POS_JOSA:
                            preferred = 1
                        var ha = Int(si[12 * slots + prev])
                        if count == 2 and (
                            first_pos == POS_NOUN or first_pos == POS_VERB_PREFIX
                        ) and pos == POS_VERB and (
                            chars[start] == 0xD558 or chars[start] == 0xD574
                        ):
                            ha = 1
                        var mismatch = Int(si[13 * slots + prev])
                        var prev_pos = Int(si[15 * slots + prev])
                        if prev_pos == POS_NOUN and pos == POS_JOSA:
                            if josa_mismatch(
                                chars, Int(si[18 * slots + prev]), start, end
                            ):
                                mismatch = 1
                        var freq_sum = sf[slots + prev] + (
                            1.0 - noun_freq if pos == POS_NOUN else 1.0
                        )
                        var words = Int(si[2 * slots + prev]) + word_inc
                        var score = candidate_score(
                            count, unknown_count, words, unknown_cov, freq_sum,
                            non_noun, det, excl, initial, preferred, ha, mismatch
                        )
                        var insert = -1
                        for rank in range(TOP_N):
                            var dst = end * TOP_N + rank
                            if score < sf[dst] or (
                                score == sf[dst] and tie < Int(si[6 * slots + dst])
                            ):
                                insert = rank
                                break
                        if insert < 0:
                            continue
                        var rank = TOP_N - 1
                        while rank > insert:
                            var dst = end * TOP_N + rank
                            var src = dst - 1
                            for field in range(NF):
                                si[field * slots + dst] = si[field * slots + src]
                            sf[dst] = sf[src]
                            sf[slots + dst] = sf[slots + src]
                            rank -= 1
                        var dst = end * TOP_N + insert
                        si[dst] = 1
                        si[slots + dst] = Int64(next_state)
                        si[2 * slots + dst] = Int64(words)
                        si[3 * slots + dst] = Int64(count)
                        si[4 * slots + dst] = Int64(unknown_count)
                        si[5 * slots + dst] = Int64(unknown_cov)
                        si[6 * slots + dst] = Int64(tie)
                        si[7 * slots + dst] = Int64(non_noun)
                        si[8 * slots + dst] = Int64(det)
                        si[9 * slots + dst] = Int64(excl)
                        si[10 * slots + dst] = Int64(initial)
                        si[11 * slots + dst] = Int64(preferred)
                        si[12 * slots + dst] = Int64(ha)
                        si[13 * slots + dst] = Int64(mismatch)
                        si[14 * slots + dst] = Int64(first_pos)
                        si[15 * slots + dst] = Int64(pos)
                        si[16 * slots + dst] = Int64(prev)
                        si[17 * slots + dst] = Int64(start)
                        si[18 * slots + dst] = Int64(end)
                        si[19 * slots + dst] = Int64(pos)
                        si[20 * slots + dst] = 1 if is_unknown else 0
                        sf[dst] = score
                        sf[slots + dst] = freq_sum
            start -= 1

    var best = n * TOP_N
    if si[best] == 0:
        starts[0] = 0
        ends[0] = Int64(n)
        poses[0] = POS_NOUN
        unknowns[0] = 1
        return 1
    var count = Int(si[3 * slots + best])
    var cursor = best
    var index = count - 1
    while index >= 0:
        starts[index] = si[17 * slots + cursor]
        ends[index] = si[18 * slots + cursor]
        poses[index] = si[19 * slots + cursor]
        unknowns[index] = si[20 * slots + cursor]
        cursor = Int(si[16 * slots + cursor])
        index -= 1
    return count
