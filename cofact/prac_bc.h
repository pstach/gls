/*
 * prac_bc.h
 *
 *  Created on: Aug 25, 2013
 *      Author: pstach
 */

#ifndef PRAC_BC_H_
#define PRAC_BC_H_

/* One more than the highest code number the byte code generator can produce */
#define BC_MAXCODE 32

#define PRAC_NR_MULTIPLIERS 19

typedef u_int8_t literal_t;
typedef u_int8_t code_t;

/* A dictionary. It contains nr_entries entries, each entry consisting of
 * a sequence of literals (a key) and a code.
 * Keys are translated to codes greedily (i.e., at any point we try to get
 * the longest dictionary match before writing a code).
 * A code of 0 in the dictionary is special and means: write nothing
 */

typedef struct bc_dict_s {
	int nr_entries;
	size_t *key_len; /* The lengths of the literal sequences */
	literal_t **key;
	code_t *code;
} bc_dict_t;

typedef struct bc_state_s {
	size_t histlen, nrstored; /* Max window size, nr of literals in history */
	literal_t *history; /* Current history window */
	code_t *buffer; /* The codes written so far */
	size_t bufalloc, buffull; /* Allocated number of codes and current number of codes in buffer */
	const bc_dict_t *dict; /* Pointer to the dictionary we use */
} bc_state_t;

extern bc_state_t *bytecoder_init(const bc_dict_t *dict);
extern void bytecoder_clear(bc_state_t *state);
extern void bytecoder(const literal_t c, bc_state_t *state);
extern void bytecoder_flush(bc_state_t *state);

/* Returns the number of bytes currently in the bytecoder buffer */
extern size_t bytecoder_size(const bc_state_t *state);
/* Writes all the data currently in the bytecoder buffer to the given pointer, and clears the buffer */
extern void bytecoder_read(code_t *dst, bc_state_t *state);

extern double prac_best(double *mul, const unsigned long n, const int m_parm,
		const double addcost, const double doublecost,
		const double bytecost, const double changecost,
		const bc_dict_t *dict);
extern double prac_bytecode(const unsigned long k, const double addcost,
		const double doublecost, const double bytecost,
		const double changecost, bc_state_t *state);

#endif /* PRAC_BC_H_ */
