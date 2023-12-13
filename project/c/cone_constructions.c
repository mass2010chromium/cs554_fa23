
/**
 * A term of the form
 *
 *   min c^T x
 *
 * Due to quirks you can only add one of these just combine it beforehand smh my head
 */
void solver_build_set_lin_term(solver_build* self, numeric* c_data, size_t n) {
    Vector* self_c = &self->c_data;
    // no need to (and don't) clear_free -- elements are not malloc'd.
    Vector_clear(self_c, n);
    for (size_t i = 0; i < n; ++i) {
        union {
            numeric d;
            void* v;
        } a;
        a.d = c_data[i];
        Vector_push(self_c, a.v);
    }
    self->unconstrained_dim = n;
}

/**
 * A term of the form:
 *
 *   min x^T LL^T x
 *
 *
 * Break down as follows:
 *
 *   min y
 *
 *   u = L^T x      unconstrained
 *   y <= u^T u
 *
 *
 * And the quadratic constraint becomes
 *
 *   ||  | 2I  0 | | u | - | 0 |  ||
 *   ||  |  0  1 | | y |   | 1 |  ||_2  <=  y.
 *
 *
 * A total of n+2 new variables are created (the u's),
 *   one y "for the inside" and one "for the outside".
 * It makes more sense if you draw out the full simplification (z = inside)
 *   and then notice that you can just overlap that with the existing x/y.
 *
 * Source: https://math.stackexchange.com/a/1952961
 */
void solver_build_add_quad_term(solver_build* self, Vector/*<matrix_entry*>*/* L_transpose, size_t num_vars) {
    size_t A_rows_old = self->b_data.size;
    size_t A_cols_old = self->c_data.size;

    // Place the block for L^T.
    for (size_t i = 0; i < L_transpose->size; ++i) {
        matrix_entry* e = L_transpose->elements[i];
        // offset the matrix block.
        e->r += A_rows_old;
        Vector_push(&self->A_data, e);
    }
    Vector_destroy(L_transpose);

    // Coefficient of the "outside" y.
    Vector_push(&self->c_data, (void*) 1);

    size_t r, c, v;
    // Place the block for identity * u. This defines the vector u.
    for (size_t i = 0; i < num_vars; ++i) {
        r = i+A_rows_old;
        c = i+A_cols_old+1; // offset 1 for the empty "output y" entry.
        v = -1;             // NOTE: i'm 99% sure the sign of x doesn't actually matter.
        Vector_push(&self->A_data, make_matrix_entry(r, c, v));

        Vector_push(&self->c_data, (void*) 0);  // Internal vars don't show up.
        Vector_push(&self->b_data, (void*) 0);  // Equality constraint.
    }

    // Place the blocks for the quadratic term.
    for (size_t i = 0; i < num_vars; ++i) {
        r = i+A_rows_old+num_vars;  // this block goes below the one above.
        c = i+A_cols_old+1;         // offset 1 for the empty "output y" entry.
        v = 2;
        Vector_push(&self->A_data, make_matrix_entry(r, c, v));

        Vector_push(&self->b_data, (void*) 0);  // All zeros then a -1.
    }
    // solitary "1" for the "inside y".
    r = A_rows_old+2*num_vars;
    c = num_vars+A_cols_old+1;
    v = 1;
    Vector_push(&self->A_data, make_matrix_entry(r, c, v));
    Vector_push(&self->b_data, (void*) -1);

    // The size of the cone is n+2.
    Vector_push(&self->cones, (void*) num_vars+2);
}
