/// Build equal-sized split chunks for a given axis length and count.
#[cfg(test)]
pub(super) fn equal_split(axis_len: usize, n: usize) -> Vec<usize> {
    if n == 0 {
        return vec![];
    }
    let chunk = axis_len.div_ceil(n);
    (0..n)
        .map(|i| {
            let start = i * chunk;
            (start + chunk).min(axis_len).saturating_sub(start)
        })
        .filter(|&s| s > 0)
        .collect()
}

#[cfg(test)]
mod basic;
#[cfg(test)]
mod dynamic_shape;
#[cfg(test)]
mod inplace;
#[cfg(test)]
mod mixed_precision;
#[cfg(test)]
mod op_placement;
#[cfg(test)]
mod parallel;
#[cfg(test)]
mod slot_dispatch;
#[cfg(test)]
mod thread_pool;
