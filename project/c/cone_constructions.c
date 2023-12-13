
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
 * A constraint of the form
 *
 *   Ax = b
 */
void solver_build_add_linear_constraint(solver_build* self, Vector/*<matrix_entry*>*/* A, numeric* b, size_t num_vars) {
    size_t A_rows_old = self->b_data.size;
    size_t A_cols_old = self->c_data.size;

    union {
        numeric d;
        void* v;
    } a;
    for (size_t i = 0; i < A->size; ++i) {
        matrix_entry* e = A->elements[i];
        // offset the matrix block.
        e->r += A_rows_old;
        Vector_push(&self->A_data, e);
    }
    Vector_destroy(A);

    for (size_t  i = 0; i < num_vars; ++i) {
        a.d = b[i];
        Vector_push(&self->b_data, a.v);  // Equality constraints for defining t.
    }
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
 *   y >= u^T u
 *
 *
 * And the quadratic constraint becomes
 *
 *   ||  | 2I  0 | | u | - | 0 |  ||
 *   ||  |  0  1 | | y |   | 1 |  ||_2  <=  y + 1.
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

    size_t r, c;
    numeric v;
    r = A_rows_old;
    c = A_cols_old;
    v = -1;
    Vector_push(&self->A_data, make_matrix_entry(r, c, v));   // coeff for z (outside)

    union {
        numeric d;
        void* v;
    } a;
    a.d = 1;
    // Coefficient of the y.
    Vector_push(&self->c_data, a.v);

    a.d = 0;
    Vector_push(&self->b_data, a.v);  // Equality constraint for z (outside).

    // Place the block for L^T.
    for (size_t i = 0; i < L_transpose->size; ++i) {
        matrix_entry* e = L_transpose->elements[i];
        // offset the matrix block.
        e->r += A_rows_old+1;
        e->v *= -2;
        Vector_push(&self->A_data, e);
    }
    Vector_destroy(L_transpose);

    a.d = 0;
    for (size_t  i = 0; i < num_vars; ++i) {
        Vector_push(&self->b_data, a.v);  // Equality constraints for defining t.
    }

    a.d = -2;
    Vector_push(&self->b_data, a.v);  // Equality constraint for z (inside).

    r = A_rows_old + num_vars + 1;
    Vector_push(&self->A_data, make_matrix_entry(r, c, v));     // coeff for z (inside)

    // The size of the cone is n+2.
    // first entry is z.
    Vector_push(&self->cones, (void*) (num_vars+2));
}

/**
 * Add a term of the form lambda * |Kx + f|_2.
 */
void solver_build_add_l2_term(solver_build* self, Vector/*<matrix_entry*>*/* K, numeric* f, numeric lambda, size_t num_vars) {
    size_t A_rows_old = self->b_data.size;
    size_t A_cols_old = self->c_data.size;

    size_t r, c;
    numeric v;
    r = A_rows_old;
    c = A_cols_old;
    v = -1;
    Vector_push(&self->A_data, make_matrix_entry(r, c, v));

    union {
        numeric d;
        void* v;
    } a;
    a.d = 0;
    Vector_push(&self->b_data, a.v);

    a.d = lambda;
    Vector_push(&self->c_data, a.v);

    for (size_t i = 0; i < K->size; ++i) {
        matrix_entry* e = K->elements[i];
        // offset the matrix block.
        e->r += A_rows_old+1;
        e->v *= -1;
        Vector_push(&self->A_data, e);
    }
    Vector_destroy(K);

    for (size_t  i = 0; i < num_vars; ++i) {
        a.d = f[i];
        Vector_push(&self->b_data, a.v);  // Equality constraints for defining t.
    }

    // The size of the cone is n+1.
    // first entry is z.
    Vector_push(&self->cones, (void*) (num_vars+1));
}
