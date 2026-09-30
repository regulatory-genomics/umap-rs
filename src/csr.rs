//! Minimal sparse matrix types ported around Python: scipy.sparse CSR/COO.
//!
//! Graph data throughout umap-learn is float32 with int32 indices, so these
//! types are specialized to that. Ops mirror the scipy methods used by the
//! Python code: `tocoo`, `tocsr`, `transpose`, `eliminate_zeros`,
//! `sum_duplicates`, elementwise multiply, addition, `maximum`.
// The kernels mirror the numba-compiled Python structure (single-char loop
// variables, index-based loops, many flag parameters matching the Python
// signatures); pedantic lints for those are allowed module-wide.
#![allow(
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::struct_excessive_bools,
    clippy::type_complexity
)]

/// Sparse matrix in coordinate format (scipy: `coo_matrix`).
#[derive(Debug, Clone, PartialEq)]
pub struct CooMatrix {
    pub shape: (usize, usize),
    pub row: Vec<i32>,
    pub col: Vec<i32>,
    pub data: Vec<f32>,
}

impl CooMatrix {
    #[must_use]
    pub fn new(shape: (usize, usize)) -> Self {
        Self {
            shape,
            row: Vec::new(),
            col: Vec::new(),
            data: Vec::new(),
        }
    }

    #[must_use]
    pub fn from_triplets(
        shape: (usize, usize),
        row: Vec<i32>,
        col: Vec<i32>,
        data: Vec<f32>,
    ) -> Self {
        assert_eq!(row.len(), col.len());
        assert_eq!(row.len(), data.len());
        Self {
            shape,
            row,
            col,
            data,
        }
    }

    /// Python: `matrix.eliminate_zeros()` — drop explicit zero entries.
    pub fn eliminate_zeros(&mut self) {
        let mut keep = Vec::with_capacity(self.data.len());
        for (idx, &v) in self.data.iter().enumerate() {
            if v != 0.0 {
                keep.push(idx);
            }
        }
        self.row = keep.iter().map(|&i| self.row[i]).collect();
        self.col = keep.iter().map(|&i| self.col[i]).collect();
        self.data = keep.iter().map(|&i| self.data[i]).collect();
    }

    /// Python: `matrix.tocsr()` — canonical CSR with duplicates summed and
    /// indices sorted per row.
    #[must_use]
    pub fn tocsr(&self) -> CsrMatrix {
        CsrMatrix::from_coo(self)
    }

    /// Python: `matrix.transpose()`.
    #[must_use]
    pub fn transpose(&self) -> Self {
        Self {
            shape: (self.shape.1, self.shape.0),
            row: self.col.clone(),
            col: self.row.clone(),
            data: self.data.clone(),
        }
    }

    /// Number of stored entries.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.data.len()
    }
}

/// Sparse matrix in compressed sparse row format (scipy: `csr_matrix`).
#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrix {
    pub shape: (usize, usize),
    /// Row pointer, length `n_rows` + 1.
    pub indptr: Vec<usize>,
    /// Column indices, length nnz.
    pub indices: Vec<i32>,
    /// Stored values, length nnz.
    pub data: Vec<f32>,
}

impl CsrMatrix {
    /// Build canonical CSR from COO: duplicates summed, indices sorted.
    #[must_use]
    pub fn from_coo(coo: &CooMatrix) -> Self {
        let n_rows = coo.shape.0;
        // Sort triplets by (row, col).
        let mut order: Vec<usize> = (0..coo.data.len()).collect();
        order.sort_unstable_by(|&a, &b| (coo.row[a], coo.col[a]).cmp(&(coo.row[b], coo.col[b])));

        let mut indptr = vec![0usize; n_rows + 1];
        let mut indices = Vec::with_capacity(coo.data.len());
        let mut data = Vec::with_capacity(coo.data.len());

        let mut last: Option<(i32, i32)> = None;
        for &i in &order {
            let r = coo.row[i];
            let c = coo.col[i];
            let v = coo.data[i];
            match last {
                Some((lr, lc)) if lr == r && lc == c => {
                    // Sum duplicate entries.
                    let n = data.len() - 1;
                    data[n] += v;
                }
                _ => {
                    indices.push(c);
                    data.push(v);
                    last = Some((r, c));
                }
            }
            let ridx = r as usize;
            indptr[ridx + 1] = indices.len();
        }
        // Fill any unset entries between empty rows with the running nnz.
        let mut running = 0usize;
        for slot in indptr.iter_mut().take(n_rows + 1) {
            if *slot < running {
                *slot = running;
            }
            running = *slot;
        }
        Self {
            shape: coo.shape,
            indptr,
            indices,
            data,
        }
    }

    /// Python: `matrix.tocoo()`.
    #[must_use]
    pub fn tocoo(&self) -> CooMatrix {
        let mut rows = Vec::with_capacity(self.data.len());
        for i in 0..self.shape.0 {
            for _ in self.indptr[i]..self.indptr[i + 1] {
                rows.push(i as i32);
            }
        }
        CooMatrix {
            shape: self.shape,
            row: rows,
            col: self.indices.clone(),
            data: self.data.clone(),
        }
    }

    /// Python: `matrix.eliminate_zeros()`.
    pub fn eliminate_zeros(&mut self) {
        let mut indices = Vec::with_capacity(self.indices.len());
        let mut data = Vec::with_capacity(self.data.len());
        let mut new_indptr = vec![0usize; self.shape.0 + 1];
        for i in 0..self.shape.0 {
            for k in self.indptr[i]..self.indptr[i + 1] {
                if self.data[k] != 0.0 {
                    indices.push(self.indices[k]);
                    data.push(self.data[k]);
                }
            }
            new_indptr[i + 1] = indices.len();
        }
        self.indptr = new_indptr;
        self.indices = indices;
        self.data = data;
    }

    /// Python: `matrix.transpose()`.
    #[must_use]
    pub fn transpose(&self) -> Self {
        // Build transposed CSR in O(nnz) via counting sort.
        let (n_rows, n_cols) = self.shape;
        let mut indptr = vec![0usize; n_cols + 1];
        for &c in &self.indices {
            indptr[c as usize + 1] += 1;
        }
        for i in 0..n_cols {
            indptr[i + 1] += indptr[i];
        }
        let mut indices = vec![0i32; self.data.len()];
        let mut data = vec![0f32; self.data.len()];
        let mut cursor = indptr.clone();
        for i in 0..n_rows {
            for k in self.indptr[i]..self.indptr[i + 1] {
                let c = self.indices[k] as usize;
                let pos = cursor[c];
                indices[pos] = i as i32;
                data[pos] = self.data[k];
                cursor[c] += 1;
            }
        }
        Self {
            shape: (n_cols, n_rows),
            indptr,
            indices,
            data,
        }
    }

    /// Row-slice access: (indices, data) for a row.
    #[must_use]
    pub fn row_slice(&self, row: usize) -> (&[i32], &[f32]) {
        let start = self.indptr[row];
        let end = self.indptr[row + 1];
        (&self.indices[start..end], &self.data[start..end])
    }

    /// Sum of all entries in a row (Python: `matrix.sum(axis=1)` per row).
    #[must_use]
    pub fn row_sums(&self) -> Vec<f32> {
        let mut sums = vec![0f32; self.shape.0];
        for i in 0..self.shape.0 {
            sums[i] = self.data[self.indptr[i]..self.indptr[i + 1]].iter().sum();
        }
        sums
    }

    /// Python: `a.multiply(b)` elementwise (on matching sparsity: used where
    /// a and b are a matrix and its transpose).
    #[must_use]
    pub fn multiply_elementwise(&self, other: &CsrMatrix) -> CsrMatrix {
        // Sparse elementwise product via hash-map-free merge over both,
        // general approach: build result COO then canonicalize.
        let mut row = Vec::new();
        let mut col = Vec::new();
        let mut data = Vec::new();
        for i in 0..self.shape.0 {
            let (ai, ad) = self.row_slice(i);
            let (bi, bd) = other.row_slice(i);
            let (n_a, n_b) = (ai.len(), bi.len());
            let mut p = 0usize;
            let mut q = 0usize;
            while p < n_a && q < n_b {
                match ai[p].cmp(&bi[q]) {
                    std::cmp::Ordering::Less => p += 1,
                    std::cmp::Ordering::Greater => q += 1,
                    std::cmp::Ordering::Equal => {
                        row.push(i as i32);
                        col.push(ai[p]);
                        data.push(ad[p] * bd[q]);
                        p += 1;
                        q += 1;
                    }
                }
            }
        }
        let coo = CooMatrix::from_triplets(self.shape, row, col, data);
        coo.tocsr()
    }

    /// Python: `a + b` (sparse addition, union of sparsity).
    #[must_use]
    pub fn add(&self, other: &CsrMatrix) -> CsrMatrix {
        let mut row = Vec::new();
        let mut col = Vec::new();
        let mut data = Vec::new();
        for i in 0..self.shape.0 {
            let (ai, ad) = self.row_slice(i);
            let (bi, bd) = other.row_slice(i);
            let (n_a, n_b) = (ai.len(), bi.len());
            let mut p = 0usize;
            let mut q = 0usize;
            while p < n_a || q < n_b {
                let ord = match (p < n_a, q < n_b) {
                    (false, true) => std::cmp::Ordering::Greater,
                    (true, false) => std::cmp::Ordering::Less,
                    (true, true) => ai[p].cmp(&bi[q]),
                    (false, false) => unreachable!(),
                };
                match ord {
                    std::cmp::Ordering::Less => {
                        row.push(i as i32);
                        col.push(ai[p]);
                        data.push(ad[p]);
                        p += 1;
                    }
                    std::cmp::Ordering::Greater => {
                        row.push(i as i32);
                        col.push(bi[q]);
                        data.push(bd[q]);
                        q += 1;
                    }
                    std::cmp::Ordering::Equal => {
                        row.push(i as i32);
                        col.push(ai[p]);
                        data.push(ad[p] + bd[q]);
                        p += 1;
                        q += 1;
                    }
                }
            }
        }
        let coo = CooMatrix::from_triplets(self.shape, row, col, data);
        coo.tocsr()
    }

    /// Python: `a.maximum(a.transpose())` — elementwise max with transpose
    /// (used for `graph_dists` in `fuzzy_simplicial_set`).
    #[must_use]
    pub fn maximum_with_transpose(&self) -> CsrMatrix {
        let t = self.transpose();
        let mut row = Vec::new();
        let mut col = Vec::new();
        let mut data = Vec::new();
        for i in 0..self.shape.0 {
            let (ai, ad) = self.row_slice(i);
            let (bi, bd) = t.row_slice(i);
            let (n_a, n_b) = (ai.len(), bi.len());
            let mut p = 0usize;
            let mut q = 0usize;
            while p < n_a || q < n_b {
                let ord = match (p < n_a, q < n_b) {
                    (false, true) => std::cmp::Ordering::Greater,
                    (true, false) => std::cmp::Ordering::Less,
                    (true, true) => ai[p].cmp(&bi[q]),
                    (false, false) => unreachable!(),
                };
                match ord {
                    std::cmp::Ordering::Less => {
                        row.push(i as i32);
                        col.push(ai[p]);
                        data.push(ad[p]);
                        p += 1;
                    }
                    std::cmp::Ordering::Greater => {
                        row.push(i as i32);
                        col.push(bi[q]);
                        data.push(bd[q]);
                        q += 1;
                    }
                    std::cmp::Ordering::Equal => {
                        row.push(i as i32);
                        col.push(ai[p]);
                        data.push(ad[p].max(bd[q]));
                        p += 1;
                        q += 1;
                    }
                }
            }
        }
        let coo = CooMatrix::from_triplets(self.shape, row, col, data);
        coo.tocsr()
    }

    /// Scale all entries: Python `matrix * scalar`.
    #[must_use]
    pub fn scale(&self, factor: f32) -> CsrMatrix {
        let mut out = self.clone();
        for v in &mut out.data {
            *v *= factor;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coo_to_csr_sums_duplicates_and_sorts() {
        let coo = CooMatrix::from_triplets(
            (3, 3),
            vec![0, 0, 1, 0],
            vec![2, 2, 0, 1],
            vec![1.0, 2.0, 3.0, 4.0],
        );
        let csr = coo.tocsr();
        assert_eq!(csr.indptr, vec![0, 2, 3, 3]);
        assert_eq!(csr.indices, vec![1, 2, 0]);
        assert_eq!(csr.data, vec![4.0, 1.0 + 2.0, 3.0]);
    }

    #[test]
    fn transpose_roundtrip() {
        let coo =
            CooMatrix::from_triplets((2, 3), vec![0, 1, 1], vec![2, 0, 1], vec![1.0, 2.0, 3.0]);
        let csr = coo.tocsr();
        let t = csr.transpose();
        assert_eq!(t.shape, (3, 2));
        let back = t.transpose();
        assert_eq!(back, csr);
    }

    #[test]
    fn add_and_multiply() {
        let a = CooMatrix::from_triplets((2, 2), vec![0, 1], vec![0, 1], vec![2.0, 3.0]).tocsr();
        let b = CooMatrix::from_triplets((2, 2), vec![0, 1], vec![1, 1], vec![5.0, 7.0]).tocsr();
        let sum = a.add(&b);
        assert_eq!(sum.data, vec![2.0, 5.0, 10.0]);
        let prod = a.multiply_elementwise(&b);
        assert_eq!(prod.data, vec![21.0]);
    }
}
