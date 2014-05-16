	.file	"test_ul128.c"
	.text
	.type	mpz_set_ul128, @function
mpz_set_ul128:
.LFB28:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsi, %rbp
	pushq	%rbx
	.cfi_def_cfa_offset 24
	.cfi_offset 3, -24
	movq	%rdi, %rbx
	subq	$8, %rsp
	.cfi_def_cfa_offset 32
	movq	8(%rsi), %rsi
	call	__gmpz_set_ui
	movq	%rbx, %rsi
	movq	%rbx, %rdi
	movl	$64, %edx
	call	__gmpz_mul_2exp
	movq	0(%rbp), %rdx
	movq	%rbx, %rsi
	addq	$8, %rsp
	.cfi_def_cfa_offset 24
	movq	%rbx, %rdi
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	jmp	__gmpz_add_ui
	.cfi_endproc
.LFE28:
	.size	mpz_set_ul128, .-mpz_set_ul128
	.type	mod128_set, @function
mod128_set:
.LFB47:
	.cfi_startproc
	pushq	%r12
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
	pushq	%rbp
	.cfi_def_cfa_offset 24
	.cfi_offset 6, -24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset 3, -32
	movq	%rdi, %rbx
	subq	$48, %rsp
	.cfi_def_cfa_offset 80
	movq	(%rsi), %rax
	leaq	32(%rsp), %r12
	leaq	16(%rsp), %rbp
	movq	%rsi, 8(%rsp)
	movq	%rax, (%rdi)
	movq	8(%rsi), %rax
	movq	%rax, 8(%rdi)
	movq	(%rsi), %rax
	leaq	2(%rax), %rdx
	movq	%rdx, %rcx
	imulq	%rax, %rcx
	addq	$2, %rcx
	imulq	%rdx, %rcx
	movq	%rcx, %rdx
	imulq	%rax, %rdx
	addq	$2, %rdx
	imulq	%rcx, %rdx
	movq	%rdx, %rcx
	imulq	%rax, %rcx
	addq	$2, %rcx
	imulq	%rdx, %rcx
	movq	%rcx, %rdx
	imulq	%rax, %rdx
	addq	$2, %rdx
	imulq	%rcx, %rdx
	imulq	%rdx, %rax
	addq	$2, %rax
	imulq	%rdx, %rax
	movq	%rax, 16(%rdi)
	movq	%r12, %rdi
	call	__gmpz_init
	movq	%rbp, %rdi
	call	__gmpz_init
	movq	8(%rsp), %rsi
	movq	%r12, %rdi
	call	mpz_set_ul128
	movl	$256, %esi
	movq	%rbp, %rdi
	call	__gmpz_setbit
	movq	%r12, %rdx
	movq	%rbp, %rsi
	movq	%rbp, %rdi
	call	__gmpz_mod
	movl	20(%rsp), %eax
	movq	$0, 24(%rbx)
	movq	$0, 32(%rbx)
	testl	%eax, %eax
	je	.L5
	decl	%eax
	je	.L6
	movq	24(%rsp), %rax
	movq	8(%rax), %rax
	movq	%rax, 32(%rbx)
.L6:
	movq	24(%rsp), %rax
	movq	(%rax), %rax
	movq	%rax, 24(%rbx)
.L5:
	leaq	16(%rsp), %rdi
	call	__gmpz_clear
	leaq	32(%rsp), %rdi
	call	__gmpz_clear
	addq	$48, %rsp
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE47:
	.size	mod128_set, .-mod128_set
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC0:
	.string	"%Zd"
	.text
	.type	ul128_print, @function
ul128_print:
.LFB32:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rdi, %rbp
	pushq	%rbx
	.cfi_def_cfa_offset 24
	.cfi_offset 3, -24
	subq	$24, %rsp
	.cfi_def_cfa_offset 48
	movq	%rsp, %rdi
	call	__gmpz_init
	movq	8(%rbp), %rsi
	movq	%rsp, %rdi
	call	__gmpz_set_ui
	movq	%rsp, %rsi
	movq	%rsp, %rdi
	movl	$64, %edx
	call	__gmpz_mul_2exp
	movq	0(%rbp), %rdx
	movq	%rsp, %rsi
	movq	%rsp, %rdi
	call	__gmpz_add_ui
	movq	%rsp, %rsi
	movl	$.LC0, %edi
	xorl	%eax, %eax
	call	__gmp_printf
	movq	%rsp, %rdi
	call	__gmpz_clear
	addq	$24, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE32:
	.size	ul128_print, .-ul128_print
	.section	.rodata.str1.1
.LC1:
	.string	"ul128_modmul"
.LC2:
	.string	"N= "
.LC3:
	.string	"SRC1_0= %llu\n"
.LC4:
	.string	"SRC2= "
.LC5:
	.string	"T= %Zd\n"
.LC6:
	.string	"T==SRC1_0*SRC2"
.LC7:
	.string	"T_0= %llu\n"
.LC8:
	.string	"NP= %llu\n"
.LC9:
	.string	"m= %llu\n"
.LC10:
	.string	"m==(T_0*NP)%%2^64\n"
.LC11:
	.string	"M= %Zd\n"
.LC12:
	.string	"M==(m*N)"
.LC13:
	.string	"T[0] NOT NULL! 0x%016llx\n"
.LC14:
	.string	"T[3] NOT NULL AT HERE2222"
.LC15:
	.string	"T[3] NOT NULL AT HERE3333"
.LC16:
	.string	"T[0] NOT NULL AT 2"
.LC17:
	.string	"T[2] WHEE 0x%016llx\n"
	.text
	.type	ul128_modmul, @function
ul128_modmul:
.LFB44:
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	xorl	%r15d, %r15d
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	movq	%rcx, %rbp
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$88, %rsp
	.cfi_def_cfa_offset 144
	movq	%rdi, 48(%rsp)
	movl	$.LC1, %edi
	movq	%rdx, 16(%rsp)
	movq	%rsi, 40(%rsp)
	call	puts
	movl	$.LC2, %edi
	xorl	%eax, %eax
	call	printf
	movq	%rbp, %rdi
	call	ul128_print
	movl	$10, %edi
	call	putchar
	movq	40(%rsp), %rax
	movl	$.LC3, %edi
	movq	(%rax), %rsi
	xorl	%eax, %eax
	call	printf
	movl	$.LC4, %edi
	xorl	%eax, %eax
	call	printf
	movq	16(%rsp), %rdi
	call	ul128_print
	movl	$10, %edi
	call	putchar
	movq	40(%rsp), %rcx
	movq	16(%rsp), %rbx
	xorl	%esi, %esi
	xorl	%edi, %edi
#APP
# 357 "../include/ul128.h" 1
	# T = src1[0] * src2
	mov (%rbx),%rax
	mulq (%rcx)
	mov %rax,%r15
	mov %rdx,%rsi
	mov 8(%rbx),%rax
	mulq (%rcx)
	xor %rdi,%rdi
	add %rax,%rsi
	adc %rdx,%rdi
	
# 0 "" 2
#NO_APP
	leaq	64(%rsp), %rbx
	movq	%rdi, 24(%rsp)
	movq	%rsi, 32(%rsp)
	movq	%rbx, %rdi
	call	__gmpz_init
	xorl	%ecx, %ecx
	movq	%rbx, %rdi
	movq	%rcx, %rsi
	movq	%rcx, (%rsp)
	call	__gmpz_set_ui
	movl	$64, %edx
	movq	%rbx, %rsi
	movq	%rbx, %rdi
	call	__gmpz_mul_2exp
	movq	24(%rsp), %rdx
	movq	%rbx, %rsi
	movq	%rbx, %rdi
	call	__gmpz_add_ui
	movl	$64, %edx
	movq	%rbx, %rsi
	movq	%rbx, %rdi
	call	__gmpz_mul_2exp
	movq	32(%rsp), %rdx
	movq	%rbx, %rsi
	movq	%rbx, %rdi
	call	__gmpz_add_ui
	movl	$64, %edx
	movq	%rbx, %rsi
	movq	%rbx, %rdi
	call	__gmpz_mul_2exp
	movq	%r15, %rdx
	movq	%rbx, %rsi
	movq	%rbx, %rdi
	call	__gmpz_add_ui
	movq	%rbx, %rsi
	movl	$.LC5, %edi
	xorl	%eax, %eax
	call	__gmp_printf
	movq	%rbx, %rdi
	call	__gmpz_clear
	movl	$.LC6, %edi
	call	puts
	movq	%r15, %r8
	movq	%r15, %rsi
	movl	$.LC7, %edi
	imulq	16(%rbp), %r8
	xorl	%eax, %eax
	movq	%r8, 56(%rsp)
	call	printf
	movq	16(%rbp), %rsi
	movl	$.LC8, %edi
	xorl	%eax, %eax
	call	printf
	movq	56(%rsp), %rsi
	movl	$.LC9, %edi
	xorl	%eax, %eax
	xorl	%r12d, %r12d
	xorl	%r13d, %r13d
	xorl	%r14d, %r14d
	call	printf
	movl	$.LC10, %edi
	xorl	%eax, %eax
	call	printf
	movq	%rbx, %rdi
	movq	56(%rsp), %r9
#APP
# 395 "../include/ul128.h" 1
	# M = m * mod->n
	mov 0(%rbp),%rax
	mulq %r9
	mov %rax,%r12
	mov %rdx,%r13
	mov 8(%rbp),%rax
	mulq %r9
	xor %r14,%r14
	add %rax,%r13
	adc %rdx,%r14
	
# 0 "" 2
#NO_APP
	call	__gmpz_init
	movq	24(%rsp), %rsi
	movq	%rbx, %rdi
	call	__gmpz_set_ui
	movl	$64, %edx
	movq	%rbx, %rsi
	movq	%rbx, %rdi
	call	__gmpz_mul_2exp
	movq	%r13, %rdx
	movq	%rbx, %rsi
	movq	%rbx, %rdi
	call	__gmpz_add_ui
	movl	$64, %edx
	movq	%rbx, %rsi
	movq	%rbx, %rdi
	call	__gmpz_mul_2exp
	movq	%r12, %rdx
	movq	%rbx, %rsi
	movq	%rbx, %rdi
	call	__gmpz_add_ui
	movq	%rbx, %rsi
	movl	$.LC11, %edi
	xorl	%eax, %eax
	call	__gmp_printf
	movq	%rbx, %rdi
	call	__gmpz_clear
	movl	$.LC12, %edi
	call	puts
	movq	32(%rsp), %rax
	movq	24(%rsp), %rdx
	movq	(%rsp), %rcx
#APP
# 425 "../include/ul128.h" 1
	# T = T + M
	add %r12,%r15
	adc %r13,%rax
	adc %r14,%rdx
	adc $0,%rcx
	
# 0 "" 2
#NO_APP
	testq	%r15, %r15
	movq	%rdx, 24(%rsp)
	movq	%rax, 32(%rsp)
	je	.L12
	movq	%r15, %rsi
	movl	$.LC13, %edi
	xorl	%eax, %eax
	movq	%rcx, (%rsp)
	call	printf
	movq	(%rsp), %rcx
.L12:
	testq	%rcx, %rcx
	je	.L13
	movl	$.LC14, %edi
	call	puts
.L13:
	xorl	%ebx, %ebx
	movq	32(%rsp), %rcx
	movq	24(%rsp), %rsi
	movq	%rbx, %r9
	movq	%rbx, %r15
	movq	40(%rsp), %rdi
	movq	16(%rsp), %r8
#APP
# 463 "../include/ul128.h" 1
	mov (%r8),%rax
	mulq 8(%rdi)
	add %rax,%rcx
	adc %rdx,%rsi
	adc $0,%r9
	mov 8(%r8),%rax
	mulq 8(%rdi)
	add %rax,%rsi
	adc %rdx,%r9
	adc $0,%r15
	
# 0 "" 2
#NO_APP
	testq	%r15, %r15
	movq	%r9, %rbx
	je	.L14
	movl	$.LC15, %edi
	movq	%rcx, (%rsp)
	movq	%rsi, 8(%rsp)
	call	puts
	movq	8(%rsp), %rsi
	movq	(%rsp), %rcx
.L14:
	movq	%rcx, %rdx
	imulq	16(%rbp), %rdx
#APP
# 497 "../include/ul128.h" 1
	# M = m * mod->n
	mov 0(%rbp),%rax
	mulq %rdx
	mov %rax,%r12
	mov %rdx,%r13
	mov 8(%rbp),%rax
	mulq %rdx
	xor %r14,%r14
	add %rax,%r13
	adc %rdx,%r14
	
# 0 "" 2
# 538 "../include/ul128.h" 1
	add %r12,%rcx
	adc %r13,%rsi
	adc %r14,%rbx
	adc $0,%r15
	
# 0 "" 2
#NO_APP
	testq	%rcx, %rcx
	je	.L15
	movl	$.LC16, %edi
	call	puts
.L15:
	testq	%r15, %r15
	je	.L16
	movq	%r15, %rsi
	movl	$.LC17, %edi
	xorl	%eax, %eax
	call	printf
.L16:
	movq	48(%rsp), %rax
	movq	%rbx, (%rax)
	movq	%r15, 8(%rax)
	movq	8(%rbp), %rdx
	cmpq	%rdx, %r15
	ja	.L17
	jb	.L19
	cmpq	0(%rbp), %rbx
	ja	.L17
	jb	.L19
.L17:
	movq	48(%rsp), %rbx
	movq	0(%rbp), %rcx
	movq	(%rbx), %rax
#APP
# 161 "../include/ul128.h" 1
	sub %rcx,%rax
	sbb %rdx,%r15
	
# 0 "" 2
#NO_APP
	movq	%rax, (%rbx)
	movq	%r15, 8(%rbx)
.L19:
	addq	$88, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE44:
	.size	ul128_modmul, .-ul128_modmul
	.type	ul128_rand, @function
ul128_rand:
.LFB50:
	.cfi_startproc
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset 3, -16
	movq	%rdi, %rbx
	call	random
	movl	%eax, (%rbx)
	call	random
	movl	%eax, 4(%rbx)
	call	random
	movl	%eax, 8(%rbx)
	call	random
	movl	%eax, 12(%rbx)
	popq	%rbx
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE50:
	.size	ul128_rand, .-ul128_rand
	.type	ul128_divrem, @function
ul128_divrem:
.LFB37:
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	movq	%rdi, %rbp
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movq	%rsi, %rbx
	subq	$88, %rsp
	.cfi_def_cfa_offset 144
	leaq	64(%rsp), %r12
	leaq	48(%rsp), %r13
	leaq	32(%rsp), %r14
	movq	%rcx, (%rsp)
	movq	%rdx, 8(%rsp)
	leaq	16(%rsp), %r15
	movq	%r12, %rdi
	call	__gmpz_init
	movq	%r13, %rdi
	call	__gmpz_init
	movq	%r14, %rdi
	call	__gmpz_init
	movq	%r15, %rdi
	call	__gmpz_init
	movq	8(%rsp), %rdx
	movq	%r14, %rdi
	movq	%rdx, %rsi
	call	mpz_set_ul128
	movq	(%rsp), %rcx
	movq	%r15, %rdi
	movq	%rcx, %rsi
	call	mpz_set_ul128
	movq	%r15, %rcx
	movq	%r14, %rdx
	movq	%r13, %rsi
	movq	%r12, %rdi
	call	__gmpz_fdiv_qr
	movl	68(%rsp), %eax
	movq	$0, 0(%rbp)
	movq	$0, 8(%rbp)
	testl	%eax, %eax
	je	.L24
	decl	%eax
	je	.L25
	movq	72(%rsp), %rax
	movq	8(%rax), %rax
	movq	%rax, 8(%rbp)
.L25:
	movq	72(%rsp), %rax
	movq	(%rax), %rax
	movq	%rax, 0(%rbp)
.L24:
	movl	52(%rsp), %eax
	movq	$0, (%rbx)
	movq	$0, 8(%rbx)
	testl	%eax, %eax
	je	.L27
	decl	%eax
	je	.L28
	movq	56(%rsp), %rax
	movq	8(%rax), %rax
	movq	%rax, 8(%rbx)
.L28:
	movq	56(%rsp), %rax
	movq	(%rax), %rax
	movq	%rax, (%rbx)
.L27:
	leaq	64(%rsp), %rdi
	call	__gmpz_clear
	leaq	48(%rsp), %rdi
	call	__gmpz_clear
	leaq	32(%rsp), %rdi
	call	__gmpz_clear
	leaq	16(%rsp), %rdi
	call	__gmpz_clear
	addq	$88, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE37:
	.size	ul128_divrem, .-ul128_divrem
	.section	.rodata.str1.1
.LC18:
	.string	"SEED: 0x%08x\n"
.LC19:
	.string	"ul_to_montgomery failed"
.LC20:
	.string	"N= %Zd\n"
.LC21:
	.string	"A= %Zd\n"
.LC22:
	.string	"MA= %Zd\n"
.LC23:
	.string	"TST= %Zd\n"
.LC24:
	.string	"ul_to_montgomery success"
.LC25:
	.string	"A_prev= "
.LC26:
	.string	"A_post= "
.LC27:
	.string	"A_post==(A_prev * 2^128) % N\n"
.LC28:
	.string	"ul_modmul failed"
.LC29:
	.string	"C= "
.LC30:
	.string	"gC= %Zd\n"
.LC31:
	.string	"ul_modmul success"
.LC32:
	.string	"ul_modinv failed"
.LC33:
	.string	"ul_modinv success"
	.text
.globl main
	.type	main, @function
main:
.LFB52:
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	movq	%rsi, %r12
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	movl	%edi, %ebp
	xorl	%edi, %edi
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$232, %rsp
	.cfi_def_cfa_offset 288
	call	time
	movq	%rax, %r13
	xorl	%eax, %eax
	call	getpid
	decl	%ebp
	jg	.L31
	movl	%eax, %ebx
	sall	$16, %ebx
	xorl	%r13d, %ebx
	jmp	.L32
.L31:
	movq	8(%r12), %rdi
	xorl	%edx, %edx
	xorl	%esi, %esi
	call	strtoul
	movl	%eax, %ebx
.L32:
	movl	%ebx, %esi
	movl	$.LC18, %edi
	xorl	%eax, %eax
	leaq	128(%rsp), %r15
	leaq	144(%rsp), %r12
	leaq	160(%rsp), %r13
	call	printf
	movl	%ebx, %edi
	leaq	192(%rsp), %rbx
	leaq	16(%rsp), %rbp
	call	srandom
	movq	%r15, %rdi
	leaq	96(%rsp), %r14
	call	__gmpz_init
	movq	%r12, %rdi
	call	__gmpz_init
	movq	%r13, %rdi
	call	__gmpz_init
	leaq	176(%rsp), %rdi
	call	__gmpz_init
	movq	%rbx, %rdi
	call	__gmpz_init
	leaq	176(%rsp), %rdi
	movl	$128, %esi
	call	__gmpz_setbit
	leaq	24(%rbp), %rax
	movl	$0, (%rsp)
	movq	%rax, 8(%rsp)
.L53:
	movq	%rbp, %rdi
	call	ul128_rand
	movq	16(%rsp), %rax
	testq	%rax, %rax
	je	.L53
	orq	$1, %rax
	movq	%rbp, %rsi
	movq	%rbp, %rdi
	movq	%rax, 16(%rsp)
	call	mod128_set
	movq	%r14, %rdi
	call	ul128_rand
	leaq	208(%rsp), %rdi
	movq	%rbp, %rcx
	movq	%r14, %rdx
	movq	%r14, %rsi
	call	ul128_divrem
	movq	%r14, %rsi
	movq	%r15, %rdi
	call	mpz_set_ul128
	movq	%rbp, %rsi
	movq	%r13, %rdi
	call	mpz_set_ul128
	movq	8(%rsp), %rdx
	leaq	80(%rsp), %rdi
	movq	%rbp, %rcx
	movq	%r14, %rsi
	call	ul128_modmul
	leaq	80(%rsp), %rsi
	movq	%r12, %rdi
	call	mpz_set_ul128
	leaq	176(%rsp), %rdx
	movq	%r15, %rsi
	movq	%rbx, %rdi
	call	__gmpz_mul
	movq	%r13, %rdx
	movq	%rbx, %rsi
	movq	%rbx, %rdi
	call	__gmpz_mod
	movq	%r12, %rsi
	movq	%rbx, %rdi
	call	__gmpz_cmp
	testl	%eax, %eax
	je	.L34
	movl	$.LC19, %edi
	call	puts
	movq	%r13, %rsi
	movl	$.LC20, %edi
	xorl	%eax, %eax
	call	__gmp_printf
	movq	%r15, %rsi
	movl	$.LC21, %edi
	xorl	%eax, %eax
	call	__gmp_printf
	movq	%r12, %rsi
	movl	$.LC22, %edi
	xorl	%eax, %eax
	call	__gmp_printf
	movq	%rbx, %rsi
	movl	$.LC23, %edi
	jmp	.L58
.L34:
	incl	(%rsp)
	cmpl	$1000000, (%rsp)
	jne	.L53
	movq	%r15, %rdi
	leaq	208(%rsp), %rbp
	xorl	%r14d, %r14d
	call	__gmpz_clear
	movq	%r12, %rdi
	call	__gmpz_clear
	movq	%r13, %rdi
	call	__gmpz_clear
	leaq	176(%rsp), %rdi
	call	__gmpz_clear
	movq	%rbx, %rdi
	call	__gmpz_clear
	movl	$.LC24, %edi
	call	puts
	movq	%rbx, %rdi
	leaq	16(%rsp), %rbx
	call	__gmpz_init
	leaq	176(%rsp), %rdi
	call	__gmpz_init
	movq	%r13, %rdi
	leaq	96(%rsp), %r13
	call	__gmpz_init
	movq	%r12, %rdi
	leaq	80(%rsp), %r12
	call	__gmpz_init
	movq	%r15, %rdi
	leaq	192(%rsp), %r15
	call	__gmpz_init
	leaq	24(%rbx), %rax
	movq	%rax, (%rsp)
.L37:
	movq	%rbp, %rdi
	call	ul128_rand
	movq	%r12, %rdi
	call	ul128_rand
.L35:
	movq	%rbx, %rdi
	call	ul128_rand
	movq	16(%rsp), %rax
	testq	%rax, %rax
	je	.L35
	orq	$1, %rax
	movq	%rbx, %rsi
	movq	%rbx, %rdi
	movq	%rax, 16(%rsp)
	call	mod128_set
	leaq	112(%rsp), %rdi
	movq	%rbx, %rcx
	movq	%rbp, %rdx
	movq	%rbp, %rsi
	call	ul128_divrem
	leaq	112(%rsp), %rdi
	movq	%rbx, %rcx
	movq	%r12, %rdx
	movq	%r12, %rsi
	call	ul128_divrem
	movq	%rbp, %rsi
	movq	%r15, %rdi
	call	mpz_set_ul128
	leaq	176(%rsp), %rdi
	movq	%r12, %rsi
	call	mpz_set_ul128
	leaq	144(%rsp), %rdi
	movq	%rbx, %rsi
	call	mpz_set_ul128
	movl	$.LC2, %edi
	xorl	%eax, %eax
	call	printf
	movq	%rbx, %rdi
	call	ul128_print
	movl	$10, %edi
	call	putchar
	movl	$.LC25, %edi
	xorl	%eax, %eax
	call	printf
	movq	%rbp, %rdi
	call	ul128_print
	movl	$10, %edi
	call	putchar
	movq	(%rsp), %rdx
	movq	%rbx, %rcx
	movq	%rbp, %rsi
	movq	%rbp, %rdi
	call	ul128_modmul
	movl	$.LC26, %edi
	xorl	%eax, %eax
	call	printf
	movq	%rbp, %rdi
	call	ul128_print
	movl	$10, %edi
	call	putchar
	movl	$.LC27, %edi
	xorl	%eax, %eax
	call	printf
	movq	(%rsp), %rdx
	movq	%rbx, %rcx
	movq	%r12, %rsi
	movq	%r12, %rdi
	call	ul128_modmul
	movq	%rbx, %rcx
	movq	%r12, %rdx
	movq	%rbp, %rsi
	movq	%r13, %rdi
	call	ul128_modmul
	leaq	64(%rsp), %rdx
	movq	%rbx, %rcx
	movq	%r13, %rsi
	movq	%r13, %rdi
	movq	$0, 72(%rsp)
	movq	$1, 64(%rsp)
	call	ul128_modmul
	leaq	176(%rsp), %rdx
	leaq	160(%rsp), %rdi
	movq	%r15, %rsi
	call	__gmpz_mul
	leaq	160(%rsp), %rsi
	leaq	144(%rsp), %rdx
	movq	%rsi, %rdi
	call	__gmpz_mod
	leaq	128(%rsp), %rdi
	movq	%r13, %rsi
	call	mpz_set_ul128
	leaq	160(%rsp), %rsi
	leaq	128(%rsp), %rdi
	call	__gmpz_cmp
	testl	%eax, %eax
	je	.L36
	movl	$.LC28, %edi
	call	puts
	movl	$.LC29, %edi
	xorl	%eax, %eax
	call	printf
	movq	%r13, %rdi
	call	ul128_print
	movl	$10, %edi
	call	putchar
	leaq	160(%rsp), %rsi
	movl	$.LC30, %edi
.L58:
	xorl	%eax, %eax
	call	__gmp_printf
.L59:
	orl	$-1, %edi
	call	exit
.L36:
	incl	%r14d
	cmpl	$1000000, %r14d
	jne	.L37
	movq	%r15, %rdi
	leaq	16(%rsp), %rbx
	leaq	112(%rsp), %r12
	call	__gmpz_clear
	leaq	176(%rsp), %rdi
	leaq	96(%rsp), %rbp
	leaq	80(%rsp), %r15
	leaq	24(%rbx), %r14
	xorl	%r13d, %r13d
	call	__gmpz_clear
	leaq	160(%rsp), %rdi
	call	__gmpz_clear
	leaq	144(%rsp), %rdi
	call	__gmpz_clear
	leaq	128(%rsp), %rdi
	call	__gmpz_clear
	movl	$.LC31, %edi
	call	puts
.L43:
	movq	%rbx, %rdi
	call	ul128_rand
	orq	$1, 16(%rsp)
.L54:
	movq	%r12, %rdi
	call	ul128_rand
	movq	%rbx, %rcx
	movq	%r12, %rdx
	movq	%r12, %rsi
	movq	%rbp, %rdi
	call	ul128_divrem
	movq	%rbx, %rdx
	movq	%r12, %rsi
	movq	%r15, %rdi
	call	ul128_gcd
	cmpq	$0, 88(%rsp)
	jne	.L54
	movq	80(%rsp), %rax
	cmpq	$1, %rax
	ja	.L54
	testq	%rax, %rax
	je	.L54
	jmp	.L63
.L64:
	movq	96(%rsp), %rax
	cmpq	$1, %rax
	ja	.L40
	testq	%rax, %rax
	jne	.L41
.L40:
	movl	$.LC32, %edi
	call	puts
	jmp	.L59
.L65:
	movl	$.LC33, %edi
	call	puts
	addq	$232, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	xorl	%eax, %eax
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
.L63:
	.cfi_restore_state
	movq	%rbx, %rsi
	movq	%rbx, %rdi
	call	mod128_set
	movq	%rbx, %rdx
	movq	%r12, %rsi
	movq	%rbp, %rdi
	call	ul128_modinv
	movq	%rbx, %rcx
	movq	%r14, %rdx
	movq	%rbp, %rsi
	movq	%rbp, %rdi
	call	ul128_modmul
	movq	%rbx, %rcx
	movq	%r14, %rdx
	movq	%r12, %rsi
	movq	%r12, %rdi
	call	ul128_modmul
	movq	%rbx, %rcx
	movq	%r12, %rdx
	movq	%rbp, %rsi
	movq	%rbp, %rdi
	call	ul128_modmul
	leaq	64(%rsp), %rdx
	movq	%rbx, %rcx
	movq	%rbp, %rsi
	movq	%rbp, %rdi
	movq	$0, 72(%rsp)
	movq	$1, 64(%rsp)
	call	ul128_modmul
	cmpq	$0, 104(%rsp)
	jne	.L40
	jmp	.L64
.L41:
	incl	%r13d
	cmpl	$1000000, %r13d
	jne	.L43
	jmp	.L65
	.cfi_endproc
.LFE52:
	.size	main, .-main
	.ident	"GCC: (GNU) 4.4.7 20120313 (Red Hat 4.4.7-3)"
	.section	.note.GNU-stack,"",@progbits
