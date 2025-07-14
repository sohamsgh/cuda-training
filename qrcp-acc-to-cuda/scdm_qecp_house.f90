module scdm_qecp
#if defined(__MPI)
  use mpi
#endif
  use scdm_profiler
#if defined(__CUDA)
  use cudafor
  use cublas_v2
  use cusolverDn
  !use nvtx
  use openacc
#endif
  implicit none
#if defined(__SCDM_SP)
  integer, parameter :: wp = kind(1.0)
  real(wp), parameter :: two_wp = 2.0, one_wp = 1.0, zero_wp = 0.0
#if defined(__MPI)
  integer, parameter :: MPI_WP = MPI_FLOAT, MPI_2WP = MPI_2REAL
#endif
#else
  integer, parameter :: wp = kind(1.d0)
  real(wp), parameter :: two_wp = 2.d0, one_wp = 1.d0, zero_wp = 0.d0
#if defined(__MPI)
  integer, parameter :: MPI_WP = MPI_DOUBLE_PRECISION, MPI_2WP = MPI_2DOUBLE_PRECISION
#endif
#endif

  integer, parameter :: orig_offset = 1 !! N.B., the ix=iy=iz=1 is at origin so we need this offset
  integer :: nobtl, ngrid(3), ngtot, unt, lwork, info
  integer :: my_nr(2,3), my_ngtot, req, qr_buf, blk_width=16
  real(wp), allocatable :: psi_ks_t(:,:), psi_ks_t_c(:,:), a(:,:), phi_scdm(:,:)
  integer, allocatable :: jpvt_loc(:), jpvt_rlv(:), jpvt(:), ig_mx_glb_mat(:,:), ig_mx_vec(:), sort_index(:)
  real(wp), allocatable :: cln_nrm(:), all_cln_nrms(:) !! column norms
  real(wp), allocatable :: reflector_cpu(:), reflector_gpu(:), svd_work_cpu(:), sing_vals(:), workarr1(:,:), workarr2(:,:) !! work arrays
  real(wp), allocatable :: u(:,:) !! output orbitals and unitary matrix to produce them
  real(wp), allocatable :: sum_rho(:), sum_s_rho(:,:), swfc(:,:) !! wave function center in crystal coordinates
  real(wp), allocatable :: spr_li(:,:) !! spreads along each lattice direction (in linear unit of each lattice vector length)
  real(wp)              :: lwork_opt(1)
  character(len=80)    :: manual_trigger_file = 'scdm_qecp_manual_jpvt_rlv.dat'
  logical              :: scdm_active = .true.
  !
  ! mpi variables
  integer, parameter :: ROOT = 0
  integer :: n_prc=1, my_rank=0,  my_prc=1, rk_with_pv=0, my_ngtot_off=0, svd_rank=0
  integer :: ictxt, nprow, npcol, myrow, mycol, llda
  integer :: desca(9),descu(9),descvt(9),m,n,mb,nb,rsrc,csrc,ia,ja
  real(wp) :: my_mx_and_rk(2), mx_and_rk(2), work_(1)
  logical :: xtrn_mpi_init = .false. !! MPI initialized in external routines
  logical :: is_root = .true.
  integer :: ierr, world_grp, comm_scdm, comm_svd
  integer, external :: numroc
  logical :: eval_spreads = .false.
  real(wp), allocatable :: rho_max_prc(:), rho_mx_vec(:)
  real(wp), allocatable :: psvd_buf(:,:), psvd_u(:,:), psvd_vt(:,:)
  !
  ! local variables
  integer :: iobtl, ix, iy, iz, ig, il
  real(wp) :: rho, sum_rho_, sum_s_rho_(3), sum_ds2_rho_(3), ds(3)
  !
  ! QR screening variables
  integer :: redlen, screening_threshold=1, expanded_pv=0
  integer, parameter :: toppercent = 2
  integer, allocatable :: rev_index(:)
  real(wp), allocatable :: cln_nrm_sort(:), store_reflectors(:,:)
  real(wp) :: normeps_vec(6)
  ! debug related
  logical :: debug = .false.
  integer, parameter :: NO_PRINT = 0
  integer, parameter :: DUMP_INPUT_PARAMS = 1
  integer, parameter :: DUMP_PIVOTS = 2
  integer, parameter :: DUMP_SPREADS = 3
  integer, parameter :: DUMP_ORBITALS = 4
  integer :: mode_debug = NO_PRINT
  logical :: debug_pv = .false.
  logical :: debug_check_pivots = .false.
  logical :: debug_check_spreads = .false.
  character(len=5) :: ciprc, cnprc
  ! other expert parameters
  integer :: screen_cols = 3 ! 0 runs QR on full wavefunction matrix, 1 screens small norms with no checks, 2 semi-robustness check, 3 guarantees same answer as 0
  logical :: use_concise_timers=.false. ! print component times in single column for easier performance analysis
  ! timing
  type(prf) :: clock_qr, clock_copy, clock_ortho, clock_u, clock_handle
  type(prf) :: clock_center, clock_spread, clock_scdm, clock_screen, clock_expand
  !
#if defined(__CUDA)
  !device related variables
  INTEGER :: ngpus
  real(wp), allocatable  :: c(:) !! c vector inside householder
  integer, value :: devicenum
  integer(acc_device_kind), value :: devicetype
  ! cublas and cusolver status variables
  integer :: istat, istat_cublas, istat_cusolver
  ! cublas and cusolver library handles
  type(cublasHandle)        :: handle_cublas
  type(cusolverDnHandle)    :: handle_cusolver
  logical :: debug_gpu = .true.
  real(wp), allocatable,device		:: svd_work_gpu(:), svd_rwork(:)
  integer :: svd_lwork
  ! create host and device pointers
  !$acc declare create(psi_ks_t, c, psi_ks_t_c,jpvt_loc,jpvt_rlv,workarr1,workarr2, store_reflectors)
  !$acc declare create(cln_nrm,cln_nrm_sort,rev_index,u,sing_vals) 
  !$acc declare create(swfc,spr_li)
  !$acc declare create(sum_s_rho,sum_s_rho_,ds,sum_rho,ig_mx_glb_mat)
  !$acc declare create(ig_mx_vec,rho_mx_vec)
  ! Following arrays are very small, maybe there is a better way?
  !$acc declare create(reflector_gpu, my_mx_and_rk, mx_and_rk, normeps_vec)
#endif
contains

  subroutine scdm_gemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    implicit none
    character(1), intent(in) :: transa, transb
    integer, intent(in) :: m, n, k, lda, ldb, ldc
    real(wp), intent(in) :: a, b, c, alpha, beta
#if defined(__SCDM_SP)
    call sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#else
    call dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
  end subroutine scdm_gemm

  subroutine scdm_larf(side, m, n, v, incv, tau, c, ldc, work)
    implicit none
    character(1), intent(in) :: side
    integer, intent(in) :: m, n, incv, ldc
    real(wp), intent(in) :: v(:), c, work(:), tau
#if defined(__SCDM_SP)
    call slarf(side, m, n, v, incv, tau, c, ldc, work)
#else
    call dlarf(side, m, n, v, incv, tau, c, ldc, work)
#endif
  end subroutine scdm_larf

  subroutine scdm_gesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info)
    implicit none
    character(1), intent(in) :: jobu, jobvt
    integer, intent(in) :: m, n, lda, ldu, ldvt, lwork, info
    real(wp) :: a(:,:), s(:), u(:,:), vt(:,:), work(:)
#if defined(__SCDM_SP)
    call sgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info)
#else
    call dgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info)
#endif    
  end subroutine scdm_gesvd

#if defined(__SCDM_SP)
  REAL function scdm_int2wp(x)
    implicit none
    integer :: x
    scdm_int2wp = real(x)
  end function scdm_int2wp
#else 
  DOUBLE PRECISION function scdm_int2wp(x)
    implicit none
    integer :: x
    scdm_int2wp = dble(x)
  end function scdm_int2wp
#endif

  subroutine  scdm_qecp_init(nobtl_,ngrid_,my_nr_,     &
      &                      eval_spreads_,use_cholesky_,debug_)
    !! [level 1] initialize scdm module (the new interface for QE)
    !! use_cholesky_ option currently defunct
    implicit none
    integer, intent(in) :: nobtl_ ! number of occupied orbitals
    integer, intent(in) :: ngrid_(3) ! (x, y, z) total grid points
    integer, intent(in) :: my_nr_(2,3) ! ((x1 x2) (y1 y2) (z1 z2))
    logical, intent(in), optional :: eval_spreads_ 
    logical, intent(in), optional :: use_cholesky_ !FIXME
    logical, intent(in), optional :: debug_
    call clock_scdm%tic
    nobtl = nobtl_
    ngrid = ngrid_
    my_nr = my_nr_
    if (present(eval_spreads_)) eval_spreads = eval_spreads_
    !if (present(use_cholesky_)) use_cholesky = use_cholesky_
    !if (present(debug_)) mode_debug = DUMP_FORCES
    if(present(debug_)) debug=debug_
    call setup_mpi
    call scan_for_expert_parameters
#if defined(__CUDA)
    call setup_gpu
#endif
    call setup_grid
    call alloc_vars
    write(ciprc,'(I0.5)') my_prc
    write(cnprc,'(I0.5)') n_prc
    call clock_scdm%toc
  end subroutine scdm_qecp_init

  subroutine  scan_for_expert_parameters()
    implicit none
    character(len=80), parameter :: trigger_file="MANUAL_SCDM_EXPERT_INPUT"
    logical :: has_expert_input
    integer :: unt, ios
    namelist /scdm_input/ mode_debug, debug_check_pivots, debug_check_spreads, screen_cols, use_concise_timers
    !
    if(is_root) then
      inquire(file=trim(trigger_file),exist=has_expert_input)
      if (has_expert_input) then
        open(newunit=unt,file=trim(trigger_file),action="read")
        read(unt,nml=scdm_input,iostat=ios)
        close(unt)
      end if
    end if
#if defined(__MPI)
    call MPI_BCAST(mode_debug,1,MPI_INTEGER,ROOT,comm_scdm,ierr)
    call MPI_BCAST(screen_cols,1,MPI_INTEGER,ROOT,comm_scdm,ierr)
#endif
  end subroutine scan_for_expert_parameters

  subroutine  setup_mpi()
    implicit none
#if defined(__MPI)
    call MPI_INITIALIZED(xtrn_mpi_init, ierr)
    if (.not.xtrn_mpi_init) call MPI_Init(ierr)
    call MPI_Comm_Group(MPI_COMM_WORLD,  world_grp, ierr)
    call MPI_Comm_Create(MPI_COMM_WORLD, world_grp, comm_scdm, ierr)
    call MPI_COMM_SIZE(comm_scdm, n_prc, ierr)
    call MPI_COMM_RANK(comm_scdm, my_rank, ierr)
#else
    my_rank = ROOT
#endif
    is_root = (my_rank.eq.ROOT)
    my_prc = my_rank + 1 ! shift to start from 1 instead of 0
  end subroutine setup_mpi

  subroutine  setup_grid()
    implicit none
    ngtot = product(ngrid)
    my_ngtot = ngrid(1)*(my_nr(2,2) - my_nr(1,2)+1)*(my_nr(2,3)-my_nr(1,3)+1) ! gridpoints handled by this process; columns of psi_ks_t input
#if defined(__MPI)
    my_ngtot_off = ngrid(1)*ngrid(2)*(my_nr(1,3)-1) + (my_nr(1,2)-1)*ngrid(1)
#endif
  end subroutine setup_grid

  subroutine  alloc_vars()
    !! [level 2] allocate variables
    implicit none
    qr_buf = my_ngtot + modulo(my_ngtot, blk_width) ! pad QR buffers to apply Householder reflectors to blocks
    call alloc_orbitals
    call alloc_pivots
    call alloc_overlap_and_orthogonalization
    call alloc_aux_arrays
  end subroutine alloc_vars

  subroutine  alloc_orbitals()
    !! [level 3] allocate orbital-related variables
    implicit none
    if (.not.allocated(psi_ks_t))    allocate(psi_ks_t(nobtl,qr_buf)) ! input/output orbitals
    if (.not.allocated(phi_scdm))    allocate(phi_scdm(my_ngtot,nobtl)) ! orbitals used for centers calculation
    if (.not.allocated(psi_ks_t_c))  allocate(psi_ks_t_c(nobtl,nobtl))  ! selected columns
    if (.not.allocated(swfc))        allocate(swfc(3,nobtl))            ! orbital centers
    if (.not.allocated(sum_rho))     allocate(sum_rho(nobtl))           ! density of each orbital
    if (.not.allocated(sum_s_rho))   allocate(sum_s_rho(3,nobtl))       ! related to density, swfc = sum_s_rho/sum_rho
    return
  end subroutine alloc_orbitals

  subroutine  alloc_pivots()
    !! [level 3] allocate variables pivot related
    implicit none
    if (.not.allocated(jpvt_loc))     allocate(jpvt_loc(nobtl))         ! local index of each QR pivot/selected column
    if (.not.allocated(cln_nrm))      allocate(cln_nrm(my_ngtot))       ! 2-norm of each column of psi_ks_t
    if (.not.allocated(workarr1))     allocate(workarr1(2*nobtl,nobtl))
    if (.not.allocated(workarr2))     allocate(workarr2(nobtl,2*nobtl))
    if (is_root.and..not.allocated(all_cln_nrms))      allocate(all_cln_nrms(my_ngtot*n_prc)) ! cln_nrm of each process collected on root
    if (is_root.and..not.allocated(sort_index))      allocate(sort_index(my_ngtot*n_prc)) ! cln_nrm of each process collected on root
    if (.not.allocated(cln_nrm_sort))      allocate(cln_nrm_sort(qr_buf)) ! column norms after columns have been rearranged during screening
    if (.not.allocated(rev_index))     allocate(rev_index(my_ngtot))    ! go from index in cln_nrm_sort to index in cln_nrm
    if (.not.allocated(jpvt_rlv))     allocate(jpvt_rlv(nobtl))         ! global index of each pivot
    if (.not.allocated(store_reflectors)) allocate(store_reflectors(nobtl,nobtl)) ! in case Householder reflectors must be applied to additional columns later
#if defined(__CUDA)
    if (.not.allocated(c))            allocate(c(nobtl)) !FIXME maybe unnecessary
    if (.not.allocated(reflector_gpu))    allocate(reflector_gpu(nobtl))
#endif
    if (.not.allocated(reflector_cpu))    allocate(reflector_cpu(nobtl))
    if (.not.allocated(ig_mx_glb_mat)) allocate(ig_mx_glb_mat(3,nobtl)) ! (x,y,z) of maximum density for each localized orbital 
    if (.not.allocated(ig_mx_vec)) allocate(ig_mx_vec(nobtl)) 
    if (.not.allocated(rho_max_prc)) allocate(rho_max_prc(n_prc)) ! used to find ig_mx_glb_mat
    if (.not.allocated(rho_mx_vec)) allocate(rho_mx_vec(nobtl)) 
    return
  end subroutine alloc_pivots

  subroutine  alloc_overlap_and_orthogonalization()
    !! [level 3] allocate variables orthogonalization related
    implicit none
    if (.not.allocated(u))        allocate(u(nobtl,nobtl)) ! unitary transformation matrix = left singular vectors * right singular vectors of selected columns
    if (.not.allocated(sing_vals)) allocate(sing_vals(nobtl)) ! singular values of psi_ks_t_c
    return
  end subroutine alloc_overlap_and_orthogonalization

  subroutine  alloc_aux_arrays()
    !! [level 3] allocate auxiliary variables
    implicit none
    if (.not.allocated(a))        allocate(a(nobtl,my_ngtot)) ! copy of psi_ks_t for performing QR
    return
  end subroutine alloc_aux_arrays

  logical function is_scdm_active()
    implicit none
    is_scdm_active = scdm_active
  end function is_scdm_active

  subroutine  activate_scdm()
    implicit none
    scdm_active = .true.
    return
  end subroutine activate_scdm

  subroutine  deactivate_scdm()
    implicit none
    scdm_active = .false.
    return
  end subroutine deactivate_scdm

  subroutine  scdm_qecp_compute()
    !! [level 1] compute scdm orbitals (step 2)
    implicit none
    call clock_scdm%tic
    if (mode_debug.ge.DUMP_INPUT_PARAMS) call debug_print_parallel_settings
    ! Perform QR with column pivoting to select well-conditioned linearly independent column subset (psi_ks_t_c) of psi_ks_t
    if (user_input_pivots()) then
      call skip_qr_and_use_input_pivots
    else
#if defined(__CUDA)
      call clock_handle%tic
      call create_cuda_handles ! TODO: check if it conflicts with QE cublas handle
      call clock_handle%toc
      call clock_qr%tic
      call perform_qr_to_get_column_pivots_gpu()
#else
      call clock_qr%tic
      call perform_qr_to_get_column_pivots_cpu()
#endif
      call clock_qr%toc
    end if
    if (debug_pv.or.debug_check_pivots) then
      call check_pivots()
    endif
    if(mode_debug.ge.DUMP_PIVOTS) then
      call dump_relevant_pivots()
    endif
    screening_threshold=1 ! reset screening threshold for multiple compute calls
    
    ! Perform symmetric orthogonalization of psi_ks_t using psi_ks_t_c to form output orbitals
    call get_scdm_orbitals_u

    ! Orbital centers
      call clock_center%tic
#if defined(__CUDA)
    call compute_scdm_orbital_center_in_crystal_coordinates_gpu
#else
    call compute_scdm_orbital_center_in_crystal_coordinates_cpu
#endif
      call clock_center%toc
    ! Orbital spreads
      call clock_spread%tic
#if defined(__CUDA)
    call compute_scdm_orbital_spreads_in_lattice_dimensions_gpu
#else
    call compute_scdm_orbital_spreads_in_lattice_dimensions_cpu
#endif
      call clock_spread%toc

    if (debug_check_spreads) then
      call check_spreads()
    endif

    if(mode_debug.ge.DUMP_INPUT_PARAMS.and.is_root) then ! print timing info
      if(use_concise_timers) then 
#if defined(__CUDA)
        call clock_handle%print_timer_concise
#endif
        call clock_qr%print_timer_concise
        call clock_copy%print_timer_concise
        call clock_u%print_timer_concise
        call clock_ortho%print_timer_concise
        call clock_center%print_timer_concise
        call clock_spread%print_timer_concise
      else
#if defined(__CUDA)
        print *, "Cuda handles: "
        call clock_handle%print_timer
#endif
        print *, "QR:"
        call clock_qr%print_timer
        print *, "copy:"
        call clock_copy%print_timer
        print *, "u:"
        call clock_u%print_timer
        print *, "ortho:"
        call clock_ortho%print_timer
        print *, "center:"
        call clock_center%print_timer
        print *, "spread:"
        call clock_spread%print_timer
      endif
    end if
    call clock_scdm%toc
    return
  end subroutine scdm_qecp_compute

  subroutine perform_qr_to_get_column_pivots_cpu()
  !! [level 2] qr with column pivoting. the pivots are linearly independent. Q is not stored. 
  !! The pivoting is not explictly performed as R is not needed, only the pivots themselves.
  !! See: https://www.cs.cornell.edu/~bindel/class/cs6210-f09/lec18.pdf
    implicit none
    real(wp) :: tau
    real(wp), allocatable :: dwork(:), avec(:)
    integer :: pv_, col
    allocate(dwork(blk_width)) ! lapack work array
    allocate(avec(blk_width))  ! used for block updating column norms

    a(1:nobtl,1:my_ngtot) = psi_ks_t(1:nobtl,1:my_ngtot)

    call compute_column_norms_local_cpu()
    if(screen_cols.gt.0) then
      call global_sort_column_norms
      call screen_column_norms_cpu
    else
      redlen = my_ngtot ! if columns were not screened by norm, the length was not reduced
      cln_nrm_sort(1:my_ngtot) = cln_nrm(1:my_ngtot)
    endif

    ! QR loop
    do pv_ = 1, nobtl
      call select_pivot_from_max_trailing_column_norms_cpu(pv_)
      ! check if chosen pivot meets screening standards (screening level 3 - less than cutoff -- level 2 - less than 0.1*cutoff)
      do while((mx_and_rk(1).lt.normeps_vec(screening_threshold).and.redlen.lt.my_ngtot.and.screen_cols.gt.2) &
           & .or.(mx_and_rk(1).lt.0.1*normeps_vec(screening_threshold).and.redlen.lt.my_ngtot.and.screen_cols.gt.1))
        call expand_qr_matrix_cpu(pv_)
        call select_pivot_from_max_trailing_column_norms_cpu(pv_) ! select new pivot with more columns in consideration
        if(mode_debug.ge.DUMP_PIVOTS) print *, "new chosen norm: ", mx_and_rk(1)
      end do
      call prepare_reflector_cpu(pv_)
      call build_selected_psi(pv_, rk_with_pv)
      call apply_reflector_cpu(pv_,dwork,avec)
    enddo
    deallocate(dwork)
    deallocate(avec)
  end subroutine perform_qr_to_get_column_pivots_cpu

  subroutine  compute_column_norms_local_cpu()
  !! [level 3] compute initial 2norm of each column. Using squared norm makes later updates slightly cheaper
    implicit none
    integer :: j
    cln_nrm = zero_wp
    !$omp parallel do
    do j = 1, my_ngtot
      cln_nrm(j) = norm2(psi_ks_t(:,j))**2
    end do ! j
    !$omp end parallel do
    return
  end subroutine compute_column_norms_local_cpu

  subroutine global_sort_column_norms()
  !! [level 3] collect all column norms to root process if necessary. Sort to get screening threshold.
    use scdm_sort_utils, only : qsort
    implicit none
#if defined(__MPI)
    call MPI_GATHER(cln_nrm,my_ngtot,MPI_WP,all_cln_nrms,my_ngtot,&
          & MPI_WP,ROOT,comm_scdm,ierr)
#else
    all_cln_nrms = cln_nrm
#endif
    if(is_root) then 
      call qsort(my_ngtot*n_prc,all_cln_nrms,sort_index)
    endif
    return
  end subroutine global_sort_column_norms

  subroutine screen_column_norms_cpu()
  !! [level 3]
  ! Since there are usually many more columns than there are pivots to choose, it is prudent to run QRCP on only a subset of columns. 
  ! Column norms are non-increasing throughout the QR process. At each step, the column with largest norm is chosen as pivot.
  ! Thus columns that initially have large norms are more likely to be chosen. We apply the QR algorithm intially to only the columns
  ! with initial norms in the top 6% of all columns.
    implicit none
    integer :: j, myi, newlen
    real(wp) :: oldeps ! if not first call, don't take columns already in A
    call clock_screen%reset
    call clock_screen%tic

    newlen = (toppercent * ngtot)/400

    if(screening_threshold.eq.1) then ! if first call, set up
      redlen = 0; cln_nrm_sort = zero_wp; psi_ks_t = zero_wp; oldeps = huge(zero_wp)
      if(is_root) then
        normeps_vec(1) = all_cln_nrms(n_prc*my_ngtot - newlen)
        normeps_vec(2) = all_cln_nrms(n_prc*my_ngtot - 4*newlen)
        normeps_vec(3) = all_cln_nrms(n_prc*my_ngtot - 4*3*newlen)
        normeps_vec(4) = all_cln_nrms(n_prc*my_ngtot - 4*9*newlen)
        normeps_vec(5) = all_cln_nrms(n_prc*my_ngtot - 4*27*newlen)
        normeps_vec(6) = -one_wp
      endif
#if defined(__MPI)
      call MPI_BCAST(normeps_vec,6,MPI_WP,ROOT,comm_scdm,ierr)
#endif
    else
      oldeps = normeps_vec(screening_threshold-1)
    endif

    !$omp parallel do private(myi)
    do j = 1, my_ngtot ! fill in columns that have sufficient norm
      if(cln_nrm(j).ge.normeps_vec(screening_threshold).and.cln_nrm(j).lt.oldeps) then
        !$omp critical
        redlen = redlen + 1
        myi = redlen
        !$omp end critical
        rev_index(myi) = j
        cln_nrm_sort(myi) = cln_nrm(j)
        psi_ks_t(:,myi) = a(:,j)
      endif
    enddo
    !$omp end parallel do
    call clock_screen%toc
    if(mode_debug.ge.DUMP_PIVOTS) print *, "min norm: ", normeps_vec(screening_threshold), "width of QR buffer: ", redlen
    if(mode_debug.ge.DUMP_INPUT_PARAMS.and.is_root) then
      print *, "Time to screen column norms: "
      call clock_screen%print_timer
    endif
  end subroutine screen_column_norms_cpu

  subroutine  select_pivot_from_max_trailing_column_norms_cpu(pv_)
  !! [level 3] note: local find max and global find max (https://www.open-mpi.org/doc/v4.1/man3/MPI_Reduce.3.php; Example 4)
    implicit none
    integer, intent(in) :: pv_
    jpvt_loc(pv_) = maxloc(cln_nrm_sort(1:redlen),dim=1)
    my_mx_and_rk = [cln_nrm_sort(jpvt_loc(pv_)), scdm_int2wp(my_rank)]
#if defined(__MPI)
    call MPI_ALLREDUCE(my_mx_and_rk, mx_and_rk, 1, MPI_2WP,&
      &                MPI_MAXLOC, comm_scdm, ierr)
    rk_with_pv = nint(mx_and_rk(2))

    if (my_rank.eq.rk_with_pv) then
      if(screen_cols.gt.0) then
        jpvt_rlv(pv_) = rev_index(jpvt_loc(pv_)) + my_ngtot_off ! transform to global id
      else
        jpvt_rlv(pv_) = jpvt_loc(pv_) + my_ngtot_off
      end if
    end if
    call MPI_BCAST(jpvt_rlv(pv_),1,MPI_INTEGER,rk_with_pv,comm_scdm,ierr)
#else
    mx_and_rk = my_mx_and_rk
    if(screen_cols.gt.0) then
      jpvt_rlv(pv_) = rev_index(jpvt_loc(pv_))
    else
      jpvt_rlv(pv_) = jpvt_loc(pv_)
    end if
#endif
  end subroutine select_pivot_from_max_trailing_column_norms_cpu

  subroutine  prepare_reflector_cpu(pv_)
  !! [level 3] put householder reflector in "reflector_cpu", broadcast
    implicit none
    integer, intent(in) :: pv_
    real(wp) :: Av(nobtl - pv_ + 1)
    reflector_cpu = zero_wp
    if (my_rank.eq.rk_with_pv) then
      Av = psi_ks_t(pv_:, jpvt_loc(pv_))
      reflector_cpu(1:nobtl - pv_ + 1) = Av/ (Av(1) + sign(norm2(Av), Av(1)))
      reflector_cpu(1) = one_wp
    endif
#if defined(__MPI)
    call MPI_BCAST(reflector_cpu,nobtl,MPI_WP,rk_with_pv,comm_scdm,ierr)
#endif
    store_reflectors(:,pv_) = reflector_cpu ! used if expand_qr_matrix called
    return
  end subroutine prepare_reflector_cpu

  subroutine build_selected_psi(pv_, rk_with_pv)
      implicit none
      integer, intent(in) :: pv_,rk_with_pv
      integer             :: i
      if (my_rank.eq.rk_with_pv) then
        !$acc update self(jpvt_loc)
        if(screen_cols.gt.0) then
          do i = 1,nobtl
              psi_ks_t_c(i,pv_) = a(i,rev_index(jpvt_loc(pv_)))
          end do
        else
          do i = 1,nobtl
              psi_ks_t_c(i,pv_) = a(i,jpvt_loc(pv_))
          end do
        end if
      else
        psi_ks_t_c(:,pv_) = zero_wp
      end if
  end subroutine build_selected_psi

  subroutine apply_reflector_cpu(pv_,dwork,avec)
    implicit none
    integer, intent(in) :: pv_
    real(wp), allocatable :: dwork(:), avec(:)
    real(wp) :: tau
    integer :: col
    tau = two_wp / dot_product(reflector_cpu(:nobtl-pv_+1),reflector_cpu(:nobtl-pv_+1)) 
    do col = 1, redlen, blk_width
      call scdm_larf('L', nobtl-pv_+1, blk_width, reflector_cpu, 1, tau, psi_ks_t(pv_,col), nobtl, dwork) ! perform Householder reflection on block of blk_width columns (lapack)
      ! only need to reflect the bottom nobtl-pv_+1 columns
      avec = psi_ks_t(pv_,col:col+blk_width-1)
      cln_nrm_sort(col:col+blk_width-1) = cln_nrm_sort(col:col+blk_width-1) - avec*avec ! update squared norms
    end do
  end subroutine apply_reflector_cpu

  subroutine expand_qr_matrix_cpu(pv_)
    implicit none
    integer, intent(in) :: pv_
    integer :: j, col, prev_len, myblk
    real(wp) :: tau(nobtl)
    real(wp), allocatable :: orgqr_lwork(:), orgqr_work(:)
    call clock_expand%reset
    call clock_expand%tic

    allocate(orgqr_lwork(1))

    if (mode_debug.ge.DUMP_PIVOTS.and.is_root) then
      print *, "EXPANDING ----------------------------------- pv = ", pv_
      print *, "chosen norm: ", mx_and_rk(1)
    end if
    prev_len = redlen
    screening_threshold = screening_threshold + 1
    call screen_column_norms_cpu

    if(expanded_pv.lt.pv_) then 
      workarr1 = zero_wp; tau = zero_wp
      do j = 1, pv_-1
        workarr1(j+1:nobtl,j) = store_reflectors(2:nobtl-j+1,j)
        tau(j) = two_wp / (one_wp+dot_product(workarr1(1:nobtl,j),workarr1(1:nobtl,j)))
      end do

#if defined(__SCDM_SP)
      call SORGQR(nobtl,nobtl,pv_-1,workarr1,2*nobtl,tau,orgqr_lwork,-1,info)
      allocate(orgqr_work(int(orgqr_lwork(1))))
      orgqr_work=zero_wp
      call SORGQR(nobtl,nobtl,pv_-1,workarr1,2*nobtl,tau,orgqr_work,int(orgqr_lwork(1)),info)
#else
      call DORGQR(nobtl,nobtl,pv_-1,workarr1,2*nobtl,tau,orgqr_lwork,-1,info)
      allocate(orgqr_work(int(orgqr_lwork(1))))
      orgqr_work=zero_wp
      call DORGQR(nobtl,nobtl,pv_-1,workarr1,2*nobtl,tau,orgqr_work,int(orgqr_lwork(1)),info)
#endif
    endif
    expanded_pv = pv_

    do j = prev_len + 1, redlen, 2*nobtl
      myblk = min(2*nobtl, redlen - j + 1)
      call scdm_gemm('T','N',nobtl-pv_+1,myblk,nobtl,one_wp,&
        &         workarr1(1,pv_),2*nobtl,psi_ks_t(1,j),nobtl,zero_wp,workarr2(pv_,1),nobtl)
      do ix = 1, myblk
        psi_ks_t(:,ix+j-1) = workarr2(:,ix)
        workarr2(:,ix)=zero_wp
      end do
    end do
    deallocate(orgqr_lwork)
    if(allocated(orgqr_work)) deallocate(orgqr_work)

    do j = prev_len+1, redlen
      cln_nrm_sort(j) = norm2(psi_ks_t(pv_:,j))**2
    end do

    call clock_expand%toc
    if(mode_debug.ge.DUMP_INPUT_PARAMS.and.is_root) then
      print *, "Time to expand qr matrix: "
      call clock_expand%print_timer
    end if
  end subroutine expand_qr_matrix_cpu

#if defined(__CUDA)
! GPU only subroutines

  subroutine screen_column_norms_gpu()
  !! [level 3] see cpu subroutine description
    implicit none
    integer :: j, myi, newlen
    real(wp) :: oldeps
    call clock_screen%reset
    call clock_screen%tic

    newlen = (toppercent * ngtot)/400
    if(screening_threshold.eq.1) then
      redlen = 0; cln_nrm_sort = zero_wp; oldeps = huge(zero_wp)
      !$acc data
      workarr1 = zero_wp
      !$acc parallel loop
      do j = 1,nobtl
        workarr1(j,j) = one_wp
      end do
      !$acc end parallel
      !$acc end data
      if(is_root) then
        normeps_vec(1) = all_cln_nrms(n_prc*my_ngtot - newlen)
        normeps_vec(2) = all_cln_nrms(n_prc*my_ngtot - 4*newlen)
        normeps_vec(3) = all_cln_nrms(n_prc*my_ngtot - 4*3*newlen)
        normeps_vec(4) = all_cln_nrms(n_prc*my_ngtot - 4*9*newlen)
        normeps_vec(5) = all_cln_nrms(n_prc*my_ngtot - 4*27*newlen)
        normeps_vec(6) = -one_wp
      endif
#if defined(__MPI)
      call MPI_BCAST(normeps_vec,6,MPI_WP,ROOT,comm_scdm,ierr)
#endif
    else
      oldeps = normeps_vec(screening_threshold-1)
    endif
    !$omp parallel do private(myi)
    do j = 1, my_ngtot
      if (cln_nrm(j).ge.normeps_vec(screening_threshold).and.cln_nrm(j).lt.oldeps) then
        !$omp critical
        redlen = redlen + 1
        myi = redlen
        !$omp end critical
        rev_index(myi) = j
        cln_nrm_sort(myi) = cln_nrm(j)
        psi_ks_t(:,myi) = a(:,j)
      endif
    enddo
    !$omp end parallel do
    !$acc update device(rev_index,cln_nrm_sort,psi_ks_t)

    call clock_screen%toc
    if(mode_debug.ge.DUMP_PIVOTS) print *, "min norm: ", normeps_vec(screening_threshold), "width of QR buffer: ", redlen
    if(mode_debug.ge.DUMP_INPUT_PARAMS.and.is_root) then
      print *, "Time to screen column norms: "
      call clock_screen%print_timer
      flush(6)
    endif
  end subroutine screen_column_norms_gpu

  subroutine expand_qr_matrix_gpu(pv_)
    use nvtx
    implicit none
    integer, intent(in) :: pv_
    integer :: j, prev_len, orgqr_lwork
    real(wp) :: prjnorm
    real(wp), allocatable :: tau(:)
    real(wp), allocatable,device :: orgqr_work(:)
    integer, device :: devInfo_d
    !$acc declare create(tau)
    call clock_expand%reset
    call clock_expand%tic

    allocate(tau(nobtl))

    if (mode_debug.ge.DUMP_PIVOTS.and.is_root) then
      print *, "EXPANDING ----------------------------------- pv = ", pv_
      print *, "chosen norm: ", mx_and_rk(1)
    end if
    prev_len = redlen
    screening_threshold = screening_threshold + 1
    !$acc update self(psi_ks_t,cln_nrm_sort)
    call screen_column_norms_gpu

    if(expanded_pv.lt.pv_) then 
      workarr1 = zero_wp; tau = zero_wp
      !$acc parallel loop gang
      do j = 1, pv_-1
        !$acc loop vector
        do iobtl = j+1, nobtl
          workarr1(iobtl,j) = store_reflectors(iobtl,j)
        end do
        tau(j) = store_reflectors(j,j)
      end do
      !$acc end parallel

      !$acc host_data use_device(workarr1,tau,orgqr_work)
      istat = cudaDeviceSynchronize()
#if defined(__SCDM_SP)
      istat =  cusolverDnSORGQR_buffersize(handle_cusolver,nobtl,nobtl,pv_-1,workarr1,2*nobtl,tau,orgqr_lwork)
      allocate(orgqr_work(orgqr_lwork))
      orgqr_work=zero_wp
      istat = cudaDeviceSynchronize()
      istat = cusolverDnSORGQR(handle_cusolver,nobtl,nobtl,pv_-1,workarr1,2*nobtl,tau,orgqr_work,orgqr_lwork,devInfo_d)
#else
      istat =  cusolverDnDORGQR_buffersize(handle_cusolver,nobtl,nobtl,pv_-1,workarr1,2*nobtl,tau,orgqr_lwork)
      allocate(orgqr_work(orgqr_lwork))
      orgqr_work=zero_wp
      istat = cudaDeviceSynchronize()
      istat = cusolverDnDORGQR(handle_cusolver,nobtl,nobtl,pv_-1,workarr1,2*nobtl,tau,orgqr_work,orgqr_lwork,devInfo_d)
#endif
      !$acc end host_data
    endif
    expanded_pv = pv_
    
    call nvtxStartRange("apply  matrix reflector", 2)
    call apply_matrix_reflector_gpu(handle_cublas,pv_,prev_len)
    call nvtxEndRange
    
    !$acc data
    !$acc parallel loop private(prjnorm)
    do j = prev_len+1, redlen
      prjnorm = zero_wp
      !$acc loop reduction(+:prjnorm)
      do ix = pv_, nobtl
        prjnorm = prjnorm + psi_ks_t(ix,j)**2
      end do
      !$acc end loop
      cln_nrm_sort(j)=prjnorm
    end do
    !$acc end parallel
    !$acc end data

    if(allocated(tau)) deallocate(tau)
    if(allocated(orgqr_work)) deallocate(orgqr_work)

    call clock_expand%toc
    if(mode_debug.ge.DUMP_INPUT_PARAMS.and.is_root) then
      print *, "Time to expand qr matrix: "
      call clock_expand%print_timer
    end if
  end subroutine expand_qr_matrix_gpu

  subroutine apply_matrix_reflector_gpu(handle_cublas, pv_,prev_len)
    implicit none
    type(cublasHandle), intent(in)		:: handle_cublas
    integer :: pv_, prev_len, j, myblk
    !$acc host_data use_device(psi_ks_t,workarr2,workarr1)
    do j = prev_len + 1, redlen, 2*nobtl
      myblk = min(2*nobtl, redlen - j + 1)
#if defined(__SCDM_SP)
      istat =  cublasSgemm(handle_cublas,CUBLAS_OP_N,CUBLAS_OP_N,nobtl-pv_+1,myblk,nobtl,& 
      & one_wp, workarr1(pv_,1),2*nobtl, psi_ks_t(1,j),nobtl,zero_wp,workarr2(pv_,1),nobtl)
#else
      istat =  cublasDgemm(handle_cublas,CUBLAS_OP_T,CUBLAS_OP_N,nobtl-pv_+1,myblk,nobtl,& 
              & one_wp, workarr1(1,pv_),2*nobtl, psi_ks_t(1,j),nobtl,zero_wp,workarr2(pv_,1),nobtl)
#endif
      !$acc parallel loop collapse(2)
      do iy = 1, myblk
        do ix = pv_, nobtl
          psi_ks_t(ix,iy + j - 1) = workarr2(ix,iy)
          workarr2(ix,iy)=zero_wp
        end do
      end do
      !$acc end parallel
    end do
    !$acc end host_data
  end subroutine apply_matrix_reflector_gpu

#endif

  subroutine  debug_print_parallel_settings()
    !! [level 2] prints parallel settings
    implicit none
    print *, 'scdm_qecp: ['//ciprc//'] nobtl:',nobtl
    print *, 'scdm_qecp: ['//ciprc//'] ngrid:',ngrid
    print *, 'scdm_qecp: ['//ciprc//'] my_nr:',my_nr
    print *, 'scdm_qecp: ['//ciprc//'] ngtot,my_ngtot, my_ngtot_off:',ngtot,my_ngtot, my_ngtot_off
    flush(6)
    return
  end subroutine debug_print_parallel_settings

  subroutine  dump_relevant_pivots()
    implicit none
    !$acc update self(jpvt_rlv)
    if (is_root) then
      open(newunit=unt,file='scdm_qecp_jpvt_rlv.dat')
      write(unt,*) nobtl
      write(unt,'(I12)') jpvt_rlv(1:nobtl)
      close(unt)
    end if
    return
  end subroutine dump_relevant_pivots

  subroutine check_pivots()
    implicit none
    integer :: nfile, ref_pivots(nobtl+1), ii, jj, mychk, overall
    !$acc update self(jpvt_rlv)
    if (is_root) then
      open(newunit=nfile,file= "./ref/scdm_qecp_jpvt_rlv.dat",action='read')
      read(nfile,'(I12)') ref_pivots(1:nobtl+1)
      close(nfile)
      overall = 0
      do ii = 1, nobtl
        mychk = 0
        do jj = 1, nobtl
          if(jpvt_rlv(jj) == ref_pivots(ii+1)) then
            mychk = 1
          endif
        enddo 
        if (mychk == 0) then
          print *, "bad pivot: ", ii
          overall = 1
        endif
      enddo 
      if (overall == 0) print *, "pivots match"
    endif
  end subroutine check_pivots

  subroutine check_spreads()
    implicit none
    integer :: nfile, ii, jj, mychk, overall
    real(wp) :: ref_spreads(3,nobtl), tol
    tol = 1.0d-6
    if (is_root) then
      open(newunit=nfile,file= "./ref/fort.801",action='read')
      read(nfile,'(3F15.7)') ref_spreads(1:3,1:nobtl)
      close(nfile)
      overall = 0
      do ii = 1, nobtl
        if(abs(spr_li(1,ii) - ref_spreads(1,ii)).gt.tol) then
          overall = overall + 1
        elseif(abs(spr_li(2,ii) - ref_spreads(2,ii)).gt.tol) then
          overall = overall + 1
        elseif(abs(spr_li(3,ii) - ref_spreads(3,ii)).gt.tol) then
          overall = overall + 1
        endif
      enddo 
      if (overall == 0) print *, "spreads match"
    endif
  end subroutine check_spreads

  logical function user_input_pivots()
    implicit none
    inquire(file=trim(manual_trigger_file), exist=user_input_pivots)
  end function user_input_pivots

  subroutine  skip_qr_and_use_input_pivots()
    implicit none
    integer :: nobtl_
    if (is_root) then
      open(newunit=unt,file=trim(manual_trigger_file),action='read')
      read(unt,*) nobtl_
      if (nobtl.ne.nobtl_) then
        write(*,*) 'ERROR: scdm_qecp: user pivot file corrupted. STOP!'
        stop
      end if
      read(unt,'(I12)') jpvt_rlv(1:nobtl)
      close(unt)
    end if
#if defined(__MPI)
    call MPI_BCAST(jpvt_rlv,nobtl,MPI_INTEGER,ROOT,comm_scdm,ierr)
#endif
  end subroutine skip_qr_and_use_input_pivots

  subroutine  scdm_qecp_finalize()
    !! [level 1] finalize (step 3)
    implicit none
    call clock_scdm%tic
    call scdm_qecp_dealloc_vars
#if defined(__MPI)
    call MPI_Comm_free(comm_scdm,ierr)
    call MPI_Group_free(world_grp,ierr)
#endif
    call clock_scdm%toc
    if(mode_debug.ge.DUMP_INPUT_PARAMS.and.is_root) then
      if(use_concise_timers) then 
        call clock_scdm%print_timer_concise
      else
        print *, "Total SCDM time: "
        call clock_scdm%print_timer
      end if
    end if
  end subroutine scdm_qecp_finalize

  subroutine  scdm_qecp_dealloc_vars()
    !! [level 2] deallocate variables
    implicit none
    if (allocated(spr_li))     deallocate(spr_li)
    if (allocated(swfc))       deallocate(swfc)
    if (allocated(sum_rho))    deallocate(sum_rho)
    if (allocated(sum_s_rho))  deallocate(sum_s_rho)
    if (allocated(rho_max_prc)) deallocate(rho_max_prc)
    if (allocated(rho_mx_vec)) deallocate(rho_mx_vec)
    if (allocated(u))          deallocate(u)
    if (allocated(jpvt_rlv))   deallocate(jpvt_rlv)
    if (allocated(jpvt))       deallocate(jpvt)
    if (allocated(a))          deallocate(a)
    if (allocated(psi_ks_t_c)) deallocate(psi_ks_t_c)
    if (allocated(psi_ks_t))   deallocate(psi_ks_t)
    if (allocated(phi_scdm))   deallocate(phi_scdm)
    if (allocated(jpvt_loc))     deallocate(jpvt_loc)
    if (allocated(cln_nrm))      deallocate(cln_nrm)
    if (allocated(jpvt_rlv))     deallocate(jpvt_rlv)
    if (allocated(cln_nrm_sort)) deallocate(cln_nrm_sort)
    if (allocated(rev_index))    deallocate(rev_index)
    if (allocated(store_reflectors)) deallocate(store_reflectors)
    if (allocated(svd_work_cpu))   deallocate(svd_work_cpu)
    if (allocated(sing_vals))  deallocate(sing_vals)
    if (allocated(ig_mx_glb_mat)) deallocate(ig_mx_glb_mat)
    if (allocated(ig_mx_vec)) deallocate(ig_mx_vec)
    if (allocated(reflector_cpu)) deallocate(reflector_cpu)
    if (allocated(workarr1))  deallocate(workarr1)
    if (allocated(workarr2))  deallocate(workarr2)
    if(allocated(psvd_buf)) deallocate(psvd_buf)
    if(allocated(psvd_u)) deallocate(psvd_u)
    if(allocated(psvd_vt)) deallocate(psvd_vt)
#if defined(__CUDA)
    if (allocated(reflector_gpu)) deallocate(reflector_gpu)
    if (allocated(c))         deallocate(c)
    if (allocated(svd_work_gpu))  deallocate(svd_work_gpu)
    if (allocated(svd_rwork)) deallocate(svd_rwork)
#endif
    return
  end subroutine scdm_qecp_dealloc_vars

  subroutine get_scdm_orbitals_u()
    implicit none
    call clock_u%tic
#if defined(__CUDA)
    call compute_unitary_svd_gpu(handle_cublas, handle_cusolver)
    !$acc update host(u)
#else
    call compute_unitary_svd_cpu
#endif
    call clock_u%toc

    call clock_ortho%tic
#if defined(__CUDA)
    call dgemm_phi_u_gpu(handle_cublas)
#else
    call dgemm_phi_u_cpu
#endif
#if defined(__MPI)
    if(mode_debug.ge.DUMP_INPUT_PARAMS) call MPI_BARRIER(comm_scdm,ierr)
#endif
    call clock_ortho%toc

  end subroutine get_scdm_orbitals_u

  subroutine  assemble_selected_psi_c()
    !! [level 3] collect selected rows of nonorthogonal SCDM orbitals to ROOT
    implicit none
#if defined(__MPI)
    if (is_root) then
      call MPI_REDUCE(MPI_IN_PLACE,psi_ks_t_c,nobtl*nobtl,MPI_WP,MPI_SUM,ROOT,comm_scdm,ierr)
    else
      call MPI_REDUCE(psi_ks_t_c,psi_ks_t_c,nobtl*nobtl,MPI_WP,MPI_SUM,ROOT,comm_scdm,ierr)
    endif
#endif
    !$acc update device(psi_ks_t_c)
    if (mode_debug.ge.DUMP_PIVOTS.and.is_root) then
      print *, 'scdm_qecp: selected columns printed in fort.203'
      do ix = 1, nobtl
        do iy = 1, nobtl-1
          write(203,'(F15.7)', advance="no") psi_ks_t_c(ix,iy)
        end do
        write(203, '(F15.7)') psi_ks_t_c(ix,nobtl)
      end do
    endif
    return
  end subroutine assemble_selected_psi_c

  subroutine dgemm_phi_u_cpu()
    implicit none

    if (is_active_proc()) call scdm_gemm('T','N',my_ngtot,nobtl,nobtl,one_wp,&
      &         a(1,1),nobtl, u(1,1),nobtl,zero_wp,phi_scdm(1,1),my_ngtot)

    if (mode_debug.ge.DUMP_ORBITALS) then
      unt = 500 + my_prc      
      do ig = 1, my_ngtot
        do iobtl = 1, nobtl-1
          write(unt,'(F15.7)',advance="no") psi_ks_t(iobtl,ig)
        end do ! iobtl
        write(unt,'(F15.7)') psi_ks_t(iobtl,ig)
      end do
      flush(unt)
    end if
  end subroutine dgemm_phi_u_cpu

  logical function is_active_proc()
    implicit none
    is_active_proc = (my_ngtot.gt.0)
  end function is_active_proc

#if defined(__MPI)
  subroutine init_svd_communicator(nsq)
    implicit none
    integer, intent(in) ::  nsq
    integer :: color

    if (my_rank.lt.nsq*nsq) then
      color = 1
    else
      color = MPI_UNDEFINED
    endif
    call MPI_Comm_split(comm_scdm,color,my_rank,comm_svd,ierr)
  end subroutine init_svd_communicator

  subroutine init_blacs_lib(nsq, submatrix)
    !! [level 3] initialization for BLACS mode
    implicit none
    integer, intent(in) :: nsq, submatrix

    nprow = nsq; npcol = nsq
    call BLACS_Pinfo(my_rank,nsq*nsq)
    call BLACS_Get(0,0,ictxt)
    call BLACS_Gridinit(ictxt,'C',nprow,npcol)
    call BLACS_Gridinfo(ictxt,nprow,npcol,myrow,mycol)

    if (my_rank.lt.nsq*nsq) then
      m=nobtl; n=nobtl; mb=submatrix; nb=submatrix; rsrc=0; csrc=0; ia=1; ja=1
      llda = numroc(nobtl, submatrix, myrow, rsrc, nprow)
      call descinit(desca,m,n,mb,nb,rsrc,csrc,ictxt,llda,info)
      call descinit(descu,m,n,mb,nb,rsrc,csrc,ictxt,llda,info)
      call descinit(descvt,m,n,mb,nb,rsrc,csrc,ictxt,llda,info)
    endif
    return
  end subroutine init_blacs_lib

  subroutine parallel_svd(nsq,submatrix)
    implicit none
    integer, intent(in) :: nsq, submatrix
    integer :: svd_lwork, xproc, yproc, pivot_root
    real(wp) :: svd_opt(2)
    
    if(.not.allocated(psvd_buf)) allocate(psvd_buf(submatrix,submatrix))
    if(.not.allocated(psvd_u)) allocate(psvd_u(submatrix,submatrix))
    if(.not.allocated(psvd_vt)) allocate(psvd_vt(submatrix,submatrix))
    call init_svd_communicator(nsq)
    call init_blacs_lib(nsq,submatrix)
    psvd_u=zero_wp

    call clock_copy%tic
    do xproc = 1, nsq
      do yproc = 1, nsq
        psvd_u(1:submatrix,1:submatrix) = psi_ks_t_c((xproc-1)*submatrix+1:xproc*submatrix,(yproc-1)*submatrix+1:yproc*submatrix)
        pivot_root = xproc-1+(yproc-1)*nsq
        call MPI_REDUCE(psvd_u,psvd_buf,submatrix*submatrix,MPI_WP,MPI_SUM,pivot_root,comm_scdm,ierr)
      end do
    end do
    call clock_copy%toc

    if (my_rank.lt.nsq*nsq) then
      if(.not.allocated(svd_work_cpu)) then
        svd_lwork = -1
#if defined(__SCDM_SP)
        call psgesvd("V","V", nobtl, nobtl, psvd_buf, ia, ja, desca, sing_vals, psvd_u, ia, ja, &
              & descu, psvd_vt, ia, ja, descvt, svd_opt, svd_lwork, info)
#else
        call pdgesvd("V","V", nobtl, nobtl, psvd_buf, ia, ja, desca, sing_vals, psvd_u, ia, ja, &
              & descu, psvd_vt, ia, ja, descvt, svd_opt, svd_lwork, info)
#endif
        svd_lwork = int(svd_opt(1))
        allocate(svd_work_cpu(svd_lwork))
      endif

#if defined(__SCDM_SP)
      call psgesvd("V","V", nobtl, nobtl, psvd_buf, ia, ja, desca, sing_vals, &
           & psvd_u, ia, ja, descu, psvd_vt, ia, ja, descvt, svd_work_cpu, svd_lwork, info)

      call psgemm('N','N',nobtl,nobtl,nobtl,one_wp, psvd_u, ia, ja, descu, &
           & psvd_vt, ia, ja, descvt, zero_wp, psvd_buf, ia, ja, desca)
#else
      call pdgesvd("V","V", nobtl, nobtl, psvd_buf, ia, ja, desca, sing_vals, &
           & psvd_u, ia, ja, descu, psvd_vt, ia, ja, descvt, svd_work_cpu, svd_lwork, info)

      call pdgemm('N','N',nobtl,nobtl,nobtl,one_wp, psvd_u, ia, ja, descu, &
           & psvd_vt, ia, ja, descvt, zero_wp, psvd_buf, ia, ja, desca)
#endif

      if(is_root.and.(debug_pv.or.mode_debug.ge.DUMP_PIVOTS)) print *, "Condition number of phi1: ", sing_vals(1)/sing_vals(nobtl)
    else
      psvd_buf=0.d0
    endif

    if (my_rank.lt.nsq*nsq) then
      call MPI_GATHER(psvd_buf, submatrix*submatrix, MPI_WP, workarr1, submatrix*submatrix, MPI_WP,ROOT,comm_svd,ierr)
      call MPI_Comm_free(comm_svd,ierr)
    endif
    if (is_root) then
      !$omp parallel do
      do yproc = 0, nsq-1
        do xproc = 0, nsq-1
          do iy = 1, submatrix
            do ix= 1, submatrix
              ig = ix + (iy-1)*submatrix + xproc*submatrix*submatrix + yproc*nsq*submatrix*submatrix
              u(xproc*submatrix + ix, yproc*submatrix + iy) = workarr1(modulo(ig,2*nobtl),floor(dble(ig)/dble(2*nobtl))+1)
            end do
          end do
        end do
      end do
      !$omp end parallel do
    endif
  end subroutine parallel_svd
#endif

  subroutine serial_svd()
    implicit none
    integer :: svd_lwork

    call clock_copy%tic
    call assemble_selected_psi_c
    call clock_copy%toc

    svd_lwork = 5*nobtl
    if (is_root) then
      if(.not.allocated(svd_work_cpu)) allocate(svd_work_cpu(svd_lwork))
      CALL scdm_gesvd("A","A", nobtl, nobtl, psi_ks_t_c, nobtl, sing_vals, &
           & workarr1, 2*nobtl, workarr2, nobtl, svd_work_cpu, svd_lwork, info)
      call scdm_gemm('N','N',nobtl,nobtl,nobtl,one_wp, workarr1(1,1), 2*nobtl, workarr2(1,1),nobtl,zero_wp,u(1,1),nobtl)
      if(debug_pv.or.mode_debug.ge.DUMP_PIVOTS) print *, "Serial SVD, condition number of phi1: ", sing_vals(1)/sing_vals(nobtl)
    end if
  end subroutine serial_svd

  subroutine  compute_unitary_svd_cpu()
    !! [level 2] compute and save unitary matrix
    !! Phi = Psi * U
    !! U = psi_ks_t_c*inv(Rchol)
    implicit none
    integer :: nsq, submatrix

    nsq=min(16, floor(sqrt(dble(n_prc))))
    do while (modulo(nobtl,nsq).ne.0)
      nsq = nsq - 1
    end do
    submatrix = nobtl/nsq

    if(nsq.gt.1) then
#if defined(__MPI)
      call parallel_svd(nsq, submatrix)
#endif
    else
      call serial_svd
    endif

#if defined(__MPI)
    call MPI_BCAST(u,nobtl*nobtl,MPI_WP,ROOT,comm_scdm,ierr)
#endif
    
    if (mode_debug.ge.DUMP_PIVOTS.and.is_root) then
      print *, 'scdm_qecp: u printed in fort.601'
      do ix = 1, nobtl
        do iy = 1, nobtl-1
          write(601, '(F15.7)', advance="no") u(ix,iy)
        end do
        write(601, '(F15.7)') u(ix,nobtl)
      end do
      flush(601)
    end if
  end subroutine compute_unitary_svd_cpu

  subroutine  compute_scdm_orbital_center_in_crystal_coordinates_cpu()
    !! [level 2] compute and wave function center
    implicit none
    call find_grid_points_with_max_self_density_cpu

    sum_rho = zero_wp; sum_s_rho = zero_wp
    do iobtl = 1, nobtl
      sum_rho_ = zero_wp; sum_s_rho_ = zero_wp
      !$omp parallel do private(ix,iy,iz,ig,il,rho,ds) &
      !$omp&            reduction(+:sum_rho_,sum_s_rho_)
      do iz = my_nr(1,3), my_nr(2,3)
        do iy = my_nr(1,2), my_nr(2,2)
          do ix = 1, ngrid(1)
            ig = ix + (iy-1)*ngrid(1) + (iz-1)*ngrid(1)*ngrid(2)
            il = ig - my_ngtot_off
            rho = phi_scdm(il,iobtl)*phi_scdm(il,iobtl)
            sum_rho_ = sum_rho_ + rho
#if defined(__SCDM_SP)
            ds(:) = real([ix,iy,iz] - ig_mx_glb_mat(:,iobtl))/real(ngrid(:))
            ds = ds - nint(ds) ! MIC (reference to ig_mx_glb as center)
            ds = ds + real(ig_mx_glb_mat(:,iobtl)-orig_offset)/real(ngrid(:)) ! shift back to lab frame within ig_mx_glb minimum image
#else
            ds(:) = dble([ix,iy,iz] - ig_mx_glb_mat(:,iobtl))/dble(ngrid(:))
            ds = ds - nint(ds) ! MIC (reference to ig_mx_glb as center)
            ds = ds + dble(ig_mx_glb_mat(:,iobtl)-orig_offset)/dble(ngrid(:)) ! shift back to lab frame within ig_mx_glb minimum image            
#endif
            sum_s_rho_(:) = sum_s_rho_(:) + ds(:) * rho
          end do ! ix
        end do ! iy
      end do ! iz
      !$omp end parallel do
      sum_rho(iobtl) = sum_rho_
      sum_s_rho(:,iobtl) = sum_s_rho_
    end do ! iobtl

#if defined(__MPI)
    call MPI_ALLREDUCE(MPI_IN_PLACE, sum_rho, nobtl,                 &
      &                MPI_WP, MPI_SUM, comm_scdm, ierr)
    call MPI_ALLREDUCE(MPI_IN_PLACE, sum_s_rho, 3*nobtl,             &
      &                MPI_WP, MPI_SUM, comm_scdm, ierr)
#endif
    !$omp parallel do
    do iobtl = 1, nobtl
      swfc(:,iobtl) = sum_s_rho(:,iobtl)/sum_rho(iobtl)
    end do
    !$omp end parallel do
    if (mode_debug.ge.DUMP_SPREADS) then
      if (is_root) then
        print *, 'scdm_qecp: swfc printed in fort.701'
        write(701,'(3F15.7)') swfc(:,:)
        flush(701)
      end if
    end if
  end subroutine compute_scdm_orbital_center_in_crystal_coordinates_cpu

  subroutine  find_grid_points_with_max_self_density_cpu()
    !! [level 3] find the location in the grid space when self density takes maxmum
    !! @note
    !! this provide a proxy to the SCDM center needed for enforcing minimum image
    !! convention for computing the first moment (swfc)
    !! @endnote
    implicit none
    integer :: n
    real(wp) :: rho_mx_glb, rho_mx_loc, rho_mx_rcv(nobtl)
    real(wp), parameter :: eps = 10.d0*EPSILON(one_wp)
    integer :: ig_mx_loc(3)

    ig_mx_vec=0; rho_mx_vec=zero_wp
    do iobtl = 1, nobtl
      rho_mx_glb = zero_wp
      !$omp parallel do reduction(max: rho_mx_glb) private(rho)
      do n = 1, my_ngtot
        rho = phi_scdm(n,iobtl)**2
        if (rho.gt.rho_mx_glb) then
          rho_mx_glb = rho
        endif
      end do
      !$omp end parallel do

      rho_mx_vec(iobtl) = rho_mx_glb
      rho_mx_glb = rho_mx_glb - eps

      !$omp parallel do collapse(2) private(rho, ig, il)
      do iz = my_nr(1,3), my_nr(2,3)
        do iy = my_nr(1,2), my_nr(2,2)
          do ix = 1, ngrid(1)
            ig = ix + (iy-1)*ngrid(1) + (iz-1)*ngrid(1)*ngrid(2)
            il = ig - my_ngtot_off
            rho = phi_scdm(il,iobtl)**2
            if (rho.ge.rho_mx_glb) then
              ig_mx_vec(iobtl) = ig
            endif
          end do
        end do
      end do
      !$omp end parallel do

    end do ! iobtl
    
#if defined(__MPI)
    call MPI_ALLREDUCE(rho_mx_vec, rho_mx_rcv, nobtl, MPI_WP,&
      &                MPI_MAX, comm_scdm, ierr)
    !$omp parallel do
    do iobtl = 1, nobtl
      if(rho_mx_rcv(iobtl)-eps.gt.rho_mx_vec(iobtl)) then
        ig_mx_vec(iobtl) = 0
      endif
    end do
    !$omp end parallel do
    call MPI_ALLREDUCE(MPI_IN_PLACE, ig_mx_vec, nobtl, MPI_INTEGER,&
      &                MPI_MAX, comm_scdm, ierr)
#endif
    !$omp parallel do
    do iobtl = 1, nobtl
      ig_mx_glb_mat(1,iobtl) = modulo(ig_mx_vec(iobtl),ngrid(1))
      ig_mx_glb_mat(3,iobtl) = floor(scdm_int2wp(ig_mx_vec(iobtl)) &
        & /scdm_int2wp(ngrid(1)*ngrid(2))) + 1
      ig_mx_glb_mat(2,iobtl) = floor(scdm_int2wp(ig_mx_vec(iobtl)- &
        & (ig_mx_glb_mat(3,iobtl)-1)*ngrid(1)*ngrid(2))/scdm_int2wp(ngrid(1))) + 1
    enddo
    !$omp end parallel do

    return
  end subroutine find_grid_points_with_max_self_density_cpu

  subroutine  compute_scdm_orbital_spreads_in_lattice_dimensions_cpu()
    !! [level 2] compute and wave function linear spreads in units of lattice vector lengths
    implicit none
    if (.not.eval_spreads) return
    if (.not.allocated(spr_li)) allocate(spr_li(3,nobtl))
    sum_rho = zero_wp; spr_li = zero_wp
    do iobtl = 1, nobtl
      sum_rho_ = zero_wp; sum_ds2_rho_ = zero_wp
      !$omp parallel do private(ix,iy,iz,ig,il,rho,ds)   &
      !$omp&            reduction(+:sum_rho_,sum_ds2_rho_)
      do iz = my_nr(1,3), my_nr(2,3)
        do iy = my_nr(1,2), my_nr(2,2)
          do ix = 1, ngrid(1)
            ig = ix + (iy-1)*ngrid(1) + (iz-1)*ngrid(1)*ngrid(2)
            il = ig - my_ngtot_off
            rho = phi_scdm(il,iobtl)*phi_scdm(il,iobtl)
            sum_rho_ = sum_rho_ + rho
#if defined(__SCDM_SP)
            ds(:) = real([ix,iy,iz]-orig_offset)/real(ngrid(:)) - swfc(:,iobtl)
#else
            ds(:) = dble([ix,iy,iz]-orig_offset)/dble(ngrid(:)) - swfc(:,iobtl)
#endif
            ds = ds - nint(ds) ! MIC
            sum_ds2_rho_(:) = sum_ds2_rho_(:) + ds(:)*ds(:) * rho
          end do ! ix
        end do ! iy
      end do ! iz
      !$omp end parallel do
      sum_rho(iobtl) = sum_rho_
      spr_li(:,iobtl) = sum_ds2_rho_
    end do ! iobtl
#if defined(__MPI)
    call MPI_ALLREDUCE(MPI_IN_PLACE, sum_rho, nobtl,                 &
      &                MPI_WP, MPI_SUM, comm_scdm, ierr)
    call MPI_ALLREDUCE(MPI_IN_PLACE, spr_li, 3*nobtl,                &
      &                MPI_WP, MPI_SUM, comm_scdm, ierr)
#endif
    !$omp parallel do
    do iobtl = 1, nobtl
      spr_li(:,iobtl) = sqrt(spr_li(:,iobtl)/sum_rho(iobtl))
    end do
    !$omp end parallel do
    if (mode_debug.ge.DUMP_SPREADS) then
      if (is_root) then
        print *, 'scdm_qecp: spreads printed in fort.801'
        write(801,'(3F15.12)') spr_li(:,:)
      end if
    end if
    return
  end subroutine compute_scdm_orbital_spreads_in_lattice_dimensions_cpu

#if defined(__CUDA)
  subroutine setup_gpu
      implicit none
      ! map ranks to devices
      ngpus = acc_get_num_devices(acc_device_nvidia)
      if (is_root .and. debug_gpu) write(0,*) 'Number of ACC devices are: ', ngpus
      call acc_set_device_num(my_rank,acc_device_nvidia)
      if (debug_gpu ) write(0,*) "rank:", my_rank, & 
              & "The ACC device number is:",acc_get_device_num(acc_device_nvidia)
  end subroutine setup_gpu

  subroutine perform_qr_to_get_column_pivots_gpu()
    use nvtx
    !! [level 2] compute qr to get pivots
    !! A is used instead of psi_ks_t as it will be overwritten at completion
    implicit none
    integer :: pv_, index_max
    real(wp) :: maxnorm, tol
    tol = 1.0D-32

    a(1:nobtl,1:my_ngtot) = psi_ks_t(1:nobtl,1:my_ngtot); psi_ks_t_c = zero_wp
    !$acc update device(psi_ks_t,psi_ks_t_c)

    call nvtxStartRange("compute_column_norms_local", 1)
    !$acc parallel  num_gangs(1280000) vector_length(256)
    call compute_column_norms_local_gpu(nobtl, my_ngtot)
    !$acc end parallel
    call nvtxEndRange
    !$acc update self(cln_nrm)
    if(screen_cols.gt.0) then
      call global_sort_column_norms
      call screen_column_norms_gpu
    else
      redlen = my_ngtot
      cln_nrm_sort = cln_nrm
      !$acc update device(cln_nrm_sort)
    endif

    do pv_ = 1, nobtl ! stop at rank
      call select_pivot_from_max_trailing_column_norms_gpu(my_rank,pv_,maxnorm,index_max,rk_with_pv,tol)
      !$acc update self(mx_and_rk)
      do while((mx_and_rk(1).lt.normeps_vec(screening_threshold).and.redlen.lt.my_ngtot.and.screen_cols.gt.2).or.(mx_and_rk(1).lt.0.1*normeps_vec(screening_threshold).and.redlen.lt.my_ngtot.and.screen_cols.gt.1))
        call expand_qr_matrix_gpu(pv_)
        call select_pivot_from_max_trailing_column_norms_gpu(my_rank,pv_,maxnorm,index_max,rk_with_pv,tol)
        !$acc update self(mx_and_rk)
        if(mode_debug.ge.DUMP_PIVOTS) print *, "new chosen norm: ", mx_and_rk(1)
      end do
      call prepare_reflector_gpu(my_rank,pv_,index_max, rk_with_pv)
      call build_selected_psi(pv_, rk_with_pv)
      call apply_reflector_gpu(pv_,nobtl,redlen)
    end do ! pv_
    psi_ks_t(1:nobtl,1:my_ngtot)=a(1:nobtl,1:my_ngtot)
    !$acc update device(psi_ks_t)
    return
  end subroutine perform_qr_to_get_column_pivots_gpu

  subroutine compute_column_norms_local_gpu(m,n)
    use m_norm, only : dnorm2_col
    !$acc routine gang
    integer, value, intent(in) :: m,n 
    integer :: j
    !$acc loop gang worker
    do j = 1, n
      cln_nrm(j) = dnorm2_col(m,j,psi_ks_t)**2 ! initial column norms
    end do ! j
  end subroutine compute_column_norms_local_gpu

  subroutine select_pivot_from_max_trailing_column_norms_gpu(my_rank,pv_,maxnorm, &
                  & index_max,rk_with_pv,tol)
    ! note: local find max and global find max (https://www.open-mpi.org/doc/v4.1/man3/MPI_Reduce.3.php; Example 4) (Example 5?)
      integer,value, intent(in)  :: pv_,my_rank
      real(wp),value, intent(in)  :: tol 
      real(wp),intent(out)        :: maxnorm
      integer, intent(out)       :: index_max, rk_with_pv
      integer                    :: i 
      maxnorm = zero_wp
      !$acc parallel loop reduction(max: maxnorm) present(cln_nrm_sort) copyin(redlen)
      do i = 1,redlen
          maxnorm = MAX(maxnorm, cln_nrm_sort(i));
      end do
      !$acc end parallel
      !$acc data copyin(redlen,maxnorm,pv_,tol) copyout(index_max) present(cln_nrm_sort,jpvt_loc) 
      !$acc parallel loop 
      do i = 1,my_ngtot
          if(ABS(cln_nrm_sort(i) - maxnorm) .LT. tol) then  !possible race condition
              !$acc atomic write
              index_max = i
              !$acc end atomic
              !$acc atomic write
              jpvt_loc(pv_) = i
              !$acc end atomic
          end if
      end do
    !$acc end parallel
    !$acc end data
    !write(0,*)" inside select pivots - just after atomic"
#if defined(__MPI)
    !$acc data copyin(my_rank,maxnorm) present(my_mx_and_rk)
    !$acc serial
    my_mx_and_rk = [maxnorm, scdm_int2wp(my_rank)]
    !$acc end serial
    !$acc end data

    !$acc host_data use_device(my_mx_and_rk,mx_and_rk)
    call MPI_ALLREDUCE(my_mx_and_rk, mx_and_rk, 1, MPI_2WP,&
      &                MPI_MAXLOC, comm_scdm, ierr) 
    !$acc end host_data

    !$acc data copyout(rk_with_pv) copyin(index_max,pv_,my_rank,my_ngtot_off)
    !$acc serial present (mx_and_rk,jpvt_rlv,jpvt_loc,rev_index)
    rk_with_pv = nint(mx_and_rk(2))
    if (my_rank.eq.rk_with_pv) then
      if(screen_cols.gt.0) then 
        jpvt_rlv(pv_) = rev_index(jpvt_loc(pv_)) + my_ngtot_off
      else
        jpvt_rlv(pv_) = jpvt_loc(pv_) + my_ngtot_off ! transform to global id
      endif
    end if
    !$acc end serial
    !$acc end data

    !$acc host_data use_device(jpvt_rlv)
    call MPI_BCAST(jpvt_rlv(pv_),1,MPI_INTEGER,rk_with_pv,comm_scdm,ierr)  
    !$acc end host_data

#else
! non-MPI
    if(screen_cols.gt.0) then
      !$acc serial present(jpvt_rlv, jpvt_loc, rev_index, mx_and_rk)
      jpvt_rlv(pv_) = rev_index(jpvt_loc(pv_))
      !$acc end serial
    else
      !$acc serial present(jpvt_rlv, jpvt_loc, mx_and_rk)
      jpvt_rlv(pv_) = jpvt_loc(pv_)
      !$acc end serial
    end if

    !$acc data copyin(my_rank,maxnorm) present(mx_and_rk)
    !$acc serial
    mx_and_rk = [maxnorm, scdm_int2wp(my_rank)]
    !$acc end serial
    !$acc end data
#endif
  end subroutine  select_pivot_from_max_trailing_column_norms_gpu

  subroutine  prepare_reflector_gpu(my_rank,pv_,index_max,rk_with_pv)
    ! should bcast the reflector
    implicit none
    integer, value,intent(in) :: my_rank,pv_, index_max, rk_with_pv
    integer                   :: i
    !$acc data copyin(pv_,index_max,rk_with_pv) present(reflector_gpu,psi_ks_t)
    ! in cuda, could put cln_nrm(index_max) on register
    if (my_rank.eq.rk_with_pv) then 
        !$acc parallel loop
        do i = pv_, nobtl 
          reflector_gpu(i) = psi_ks_t(i,index_max)
        end do
        !$acc end parallel
    end if
    !$acc end data

#if defined(__MPI)
    !$acc host_data use_device(reflector_gpu)
    call MPI_BCAST(reflector_gpu,nobtl,MPI_WP,rk_with_pv,comm_scdm,ierr) 
    !$acc end host_data
#endif
    return
  end subroutine prepare_reflector_gpu

  SUBROUTINE apply_reflector_gpu(pv_,m,n)
    integer, value, intent(in)  :: pv_, m, n
    integer :: i,j,k,ipv,counter
    ! INTEGER n,np
    ! REAL a(np,np),c(n),d(n)
    ! LOGICAL							:: sing
    ! Constructs the QR decomposition of a(1:n,1:n), with physical dimension np. 
    ! The upper triangular matrix R is returned in the upper triangle of a, except 
    ! for the diagonal elements of R which are returned in d(1:n). The 
    ! orthogonal matrix Q is represented as a product of n−1 
    ! Householder matrices Q1 ...Qn−1, where Qj = 1−uj ⊗uj/cj. The ith component 
    ! of uj is zero for i = 1, . . . , j − 1 while the nonzero components are 
    ! returned in a(i,j) for i = j, . . . , n. sing returns as true if singularity 
    ! is encountered during the decomposition, but the decomposition 
    ! is still completed in this case.
    real(wp) :: sigma,summ,tau

    summ=zero_wp
    !$acc parallel loop reduction(+:summ) present(reflector_gpu)
    do i=pv_,m !12
      summ=summ+reflector_gpu(i)**2 
    enddo      !12 
    !$acc end parallel 

    !$acc serial present(reflector_gpu) copyout(sigma)
    sigma=reflector_gpu(pv_)+sign(sqrt(summ),reflector_gpu(pv_))
    reflector_gpu(pv_)=one_wp
    !$acc end serial

    summ = one_wp
    !$acc parallel loop reduction(+:summ) present(reflector_gpu) copyin(sigma)
    do i = pv_+1,m
      reflector_gpu(i) = reflector_gpu(i)/sigma
      store_reflectors(i,pv_) = reflector_gpu(i)
      summ = summ + reflector_gpu(i)**2
    end do
    !$acc end parallel

    tau = two_wp / summ
    !$acc serial copyin(tau)
    store_reflectors(pv_,pv_) = tau
    !$acc end serial

    !$acc parallel loop gang present(reflector_gpu,psi_ks_t,cln_nrm_sort) 
    do j=1,n !16
      summ=zero_wp
      !$acc loop vector reduction(+:summ)
      do i=pv_,m  !14
        summ=summ+reflector_gpu(i)*psi_ks_t(i,j) 
      enddo       !14
      summ=summ*tau
      !$acc loop vector  
      do i=pv_,m  !15
        psi_ks_t(i,j)=psi_ks_t(i,j)-summ*reflector_gpu(i) 
      enddo       !15
      cln_nrm_sort(j) = cln_nrm_sort(j) - psi_ks_t(pv_,j)**2
    enddo   !16
    !$acc end parallel 

    return
  END SUBROUTINE apply_reflector_gpu 

  subroutine  create_cuda_handles()
    use cublas_v2
    use cusolverDn
    implicit none
    ! GPU cublas cusolver handle creation
    istat_cublas = cublasCreate(handle_cublas)
    if (istat_cublas /= CUBLAS_STATUS_SUCCESS) then
      write(0,*) 'cublas handle creation failed'
      STOP 1
    end if
    istat_cusolver = cusolverDnCreate(handle_cusolver)
    if (istat_cusolver /= CUSOLVER_STATUS_SUCCESS) then
      print*, 'cusolverDn handle creation failed'
      STOP 3
    end if
    istat_cusolver =  cusolverDnSetStream(handle_cusolver,1)
#if defined(__SCDM_SP)
    istat = cusolverDnSgesvd_bufferSize(handle_cusolver,nobtl,nobtl,svd_lwork)
#else
    istat = cusolverDnDgesvd_bufferSize(handle_cusolver,nobtl,nobtl,svd_lwork)
#endif
    if (istat /= CUSOLVER_STATUS_SUCCESS) &
        write(0,*) 'cusolverDnDpotrf_buffersize failed'
    allocate(svd_work_gpu(svd_lwork)) 
    allocate(svd_rwork(nobtl-1)) 
  end subroutine create_cuda_handles

  subroutine  compute_unitary_svd_gpu(handle_cublas, handle_cusolver)
    !! [level 2] compute and save unitary matrix
    !! Phi = Psi * U -- Phi^T = U^T * Psi^T
    !! U = left singular vectors * right singular vectors of psi_ks_t_c
    use cudafor
    implicit none
    type(cublasHandle), intent(in)		:: handle_cublas
    type(cusolverDnHandle)    :: handle_cusolver
    integer				:: info
    integer :: istat
    !note the use of device: breaks portability, but best option for now
    integer, device :: devInfo_d

    call clock_copy%tic
    call assemble_selected_psi_c
    call clock_copy%toc
    
    if (is_active_proc()) then
      istat = cudaDeviceSynchronize()
      !$acc host_data use_device(psi_ks_t_c,sing_vals,workarr1,workarr2,u)
#if defined(__SCDM_SP)
      istat = cusolverDnSgesvd(handle_cusolver, "A","A", nobtl, nobtl, psi_ks_t_c, nobtl, & 
      & sing_vals, workarr1, 2*nobtl, workarr2, nobtl, svd_work_gpu, &
      & svd_lwork, svd_rwork, devInfo_d)
#else
      istat = cusolverDnDgesvd(handle_cusolver, "A","A", nobtl, nobtl, psi_ks_t_c, nobtl, & 
                  & sing_vals, workarr1, 2*nobtl, workarr2, nobtl, svd_work_gpu, &
                  & svd_lwork, svd_rwork, devInfo_d)
#endif
      if (istat /= CUSOLVER_STATUS_SUCCESS .and. is_root) &
            write(0,*) 'cusolverDn_gesvd failed with info:',istat
      istat = devInfo_d
      if (istat /= 0 .and. is_root) then 
              write(0,*) "SVD failed wih info:",istat
      else
              write(0,*) "SVD successful"
      end if
      istat = cudaDeviceSynchronize()
#if defined(__SCDM_SP)
      istat =  cublasSgemm(handle_cublas,CUBLAS_OP_N,CUBLAS_OP_N,nobtl,nobtl,nobtl,& 
      & one_wp, workarr1,2*nobtl, workarr2,nobtl,zero_wp,u,nobtl)
#else
      istat =  cublasDgemm(handle_cublas,CUBLAS_OP_N,CUBLAS_OP_N,nobtl,nobtl,nobtl,& 
              & one_wp, workarr1,2*nobtl, workarr2,nobtl,zero_wp,u,nobtl)
#endif
      !$acc end host_data
      if(is_root.and.(debug_pv.or.mode_debug.ge.DUMP_PIVOTS)) then
          !$acc update self(sing_vals)
          write(0,*), "Condition number of phi1: ", sing_vals(1)/sing_vals(nobtl)
      end if
    end if
    if (mode_debug.ge.DUMP_PIVOTS) then
      !$acc update self(u)
      if (is_root) then
        print *, 'scdm_qecp: flattened u printed in fort.601'
        write(601,'(F15.7)') u(:,:)
        flush(601)
      end if
    end if
    return
  end subroutine compute_unitary_svd_gpu

  subroutine dgemm_phi_u_gpu(handle_cublas)
    implicit none
    type(cublasHandle), intent(in) :: handle_cublas
    integer :: pv_, prev_len, j, myblk

    if (is_active_proc()) then 
      !$acc host_data use_device(psi_ks_t,u,workarr1)
      do j = 1, my_ngtot, 2*nobtl
        myblk = min(2*nobtl, my_ngtot - j + 1)
        ! myblk = my_ngtot;j=1
#if defined(__SCDM_SP)
        istat =  cublasSgemm(handle_cublas, CUBLAS_OP_T, CUBLAS_OP_N, &
        &         myblk,nobtl,nobtl,one_wp,&
        &         psi_ks_t(1,j),nobtl,u,nobtl,zero_wp,workarr1,2*nobtl)
#else
        istat =  cublasDgemm(handle_cublas, CUBLAS_OP_T, CUBLAS_OP_N, &
        &         myblk,nobtl,nobtl,one_wp,&
        &         psi_ks_t(1,j),nobtl,u,nobtl,zero_wp,workarr1,2*nobtl)
#endif
        !$acc parallel loop collapse(2)
        do iy = 1, myblk
          do ix = 1, nobtl
            psi_ks_t(ix,iy + j - 1) = workarr1(iy,ix)
          end do
        end do
        !$acc end parallel
      end do
      !$acc end host_data

    !$acc update self(psi_ks_t)
    end if
    if (mode_debug.ge.DUMP_ORBITALS) then
      unt = 500 + my_prc      
      do ig = 1, my_ngtot
        do iobtl = 1, nobtl-1
          write(unt,'(F15.7)',advance="no") psi_ks_t(iobtl,ig)
        end do ! iobtl
        write(unt,'(F15.7)') psi_ks_t(iobtl,ig)
      end do
      flush(unt)
    end if
  end subroutine dgemm_phi_u_gpu

  subroutine  compute_scdm_orbital_center_in_crystal_coordinates_gpu()
    !! [level 2] compute and wave function center
    implicit none
    ! integer, intent(in) :: nobtl, ngtot, ngrid(3)
    real(wp), parameter :: eps = 10.d0*EPSILON(one_wp)
    real(wp) :: rho_mx_glb,sumx,sumy,sumz
    call find_grid_points_with_max_self_density_gpu

    sum_rho = zero_wp; sum_s_rho = zero_wp
    !$acc parallel loop gang private(sum_rho_, sum_s_rho_, rho_mx_glb, ig, sumx, sumy, sumz, ix, iy, iz, ds, rho)
    do iobtl = 1, nobtl
      sum_rho_ = zero_wp; sum_s_rho_ = zero_wp; sumx=zero_wp;sumy=zero_wp;sumz=zero_wp
      !$acc loop vector collapse(2) reduction(+:sumx,sumy,sumz,sum_rho_) private(rho, ig, il, ds,sumx,sumy,sumz,sum_rho_)
      do iz = my_nr(1,3), my_nr(2,3)
        do iy = my_nr(1,2), my_nr(2,2)
          do ix = 1, ngrid(1)
            ig = ix + (iy-1)*ngrid(1) + (iz-1)*ngrid(1)*ngrid(2)
            il = ig - my_ngtot_off
            rho = psi_ks_t(iobtl,il)*psi_ks_t(iobtl,il)
            sum_rho_ = sum_rho_ + rho
            ! for some reason doing this on the vector level breaks
            ds(1) = scdm_int2wp(ix - ig_mx_glb_mat(1,iobtl))/scdm_int2wp(ngrid(1))
            ds(2) = scdm_int2wp(iy - ig_mx_glb_mat(2,iobtl))/scdm_int2wp(ngrid(2))
            ds(3) = scdm_int2wp(iz - ig_mx_glb_mat(3,iobtl))/scdm_int2wp(ngrid(3))
            ds(1) = ds(1) - nint(ds(1)) ! MIC (reference to ig_mx_glb as center)
            ds(2) = ds(2) - nint(ds(2))
            ds(3) = ds(3) - nint(ds(3))
            ds(1) = ds(1) + scdm_int2wp(ig_mx_glb_mat(1,iobtl)-orig_offset)/scdm_int2wp(ngrid(1)) ! shift back to lab frame within ig_mx_glb minimum image
            ds(2) = ds(2) + scdm_int2wp(ig_mx_glb_mat(2,iobtl)-orig_offset)/scdm_int2wp(ngrid(2)) 
            ds(3) = ds(3) + scdm_int2wp(ig_mx_glb_mat(3,iobtl)-orig_offset)/scdm_int2wp(ngrid(3)) 
            sumx = sumx + ds(1) * rho
            sumy = sumy + ds(2) * rho
            sumz = sumz + ds(3) * rho
           end do ! ix
        end do ! iy
      end do ! iz
      !$acc end loop

      sum_rho(iobtl) = sum_rho_
      sum_s_rho(:,iobtl) = [sumx,sumy,sumz]
    end do ! iobtl
    !$acc end parallel loop

#if defined(__MPI)
    !$acc update self(sum_rho, sum_s_rho)
    call MPI_ALLREDUCE(MPI_IN_PLACE, sum_rho, nobtl,                 &
      &                MPI_WP, MPI_SUM, comm_scdm, ierr)
    call MPI_ALLREDUCE(MPI_IN_PLACE, sum_s_rho, 3*nobtl,             &
      &                MPI_WP, MPI_SUM, comm_scdm, ierr)
    !$acc update device(sum_rho, sum_s_rho)
#endif
  
    !$acc parallel loop
    do iobtl = 1, nobtl
      swfc(:,iobtl) = sum_s_rho(:,iobtl)/sum_rho(iobtl)
    end do
    !$acc end parallel

    !$acc update self(swfc)

    if (is_root.and.mode_debug.ge.DUMP_SPREADS) then
        print *, 'scdm_qecp: swfc printed in fort.700'
        write(700,'(3F15.7)') swfc(:,:)
        flush(700)
    end if

    return
  end subroutine compute_scdm_orbital_center_in_crystal_coordinates_gpu

  subroutine find_grid_points_with_max_self_density_gpu()
    implicit none
    real(wp) :: rho_mx_glb, rho_mx_loc, rho_mx_rcv(nobtl)
    real(wp), parameter :: eps = 10.d0*EPSILON(one_wp)
    integer :: n
    

    ig_mx_vec=0; rho_mx_vec=0
    !$acc parallel loop gang private(rho_mx_glb, ig, il, ix, iy, iz, rho, n) 
    do iobtl = 1, nobtl
      rho_mx_glb = zero_wp
      !$acc loop vector reduction(max: rho_mx_glb) private(rho, rho_mx_glb)
      do n = 1, my_ngtot
        rho = psi_ks_t(iobtl,n)**2
        if (rho.gt.rho_mx_glb) then
          rho_mx_glb = rho
        endif
      end do
      !$acc end loop

      rho_mx_vec(iobtl) = rho_mx_glb
      rho_mx_glb = rho_mx_glb - eps

      !$acc loop vector collapse(2) private(rho, ig, il) 
      do iz = my_nr(1,3), my_nr(2,3)
        do iy = my_nr(1,2), my_nr(2,2)
          do ix = 1, ngrid(1)
            ig = ix + (iy-1)*ngrid(1) + (iz-1)*ngrid(1)*ngrid(2)
            il = ig - my_ngtot_off
            rho = psi_ks_t(iobtl,il)**2
            if (rho.ge.rho_mx_glb) then
              ig_mx_vec(iobtl) = ig
            endif
          end do
        end do
      end do
      !$acc end loop
    end do ! iobtl
    !$acc end parallel loop

#if defined(__MPI)
    !$acc update self(rho_mx_vec, ig_mx_vec)
    call MPI_ALLREDUCE(rho_mx_vec, rho_mx_rcv, nobtl, MPI_WP,&
      &                MPI_MAX, comm_scdm, ierr)
    do iobtl = 1, nobtl
      if(rho_mx_rcv(iobtl)-eps.gt.rho_mx_vec(iobtl)) then
        ig_mx_vec(iobtl) = 0
      endif
    end do
    call MPI_ALLREDUCE(MPI_IN_PLACE, ig_mx_vec, nobtl, MPI_INTEGER,&
      &                MPI_MAX, comm_scdm, ierr)
#endif
    do iobtl = 1, nobtl
      ig_mx_glb_mat(1,iobtl) = modulo(ig_mx_vec(iobtl),ngrid(1))
      ig_mx_glb_mat(3,iobtl) = floor(scdm_int2wp(ig_mx_vec(iobtl)) &
        & /scdm_int2wp(ngrid(1)*ngrid(2))) + 1
      ig_mx_glb_mat(2,iobtl) = floor(scdm_int2wp(ig_mx_vec(iobtl)- &
        & (ig_mx_glb_mat(3,iobtl)-1)*ngrid(1)*ngrid(2))/scdm_int2wp(ngrid(1))) + 1
    enddo
    !$acc update device(ig_mx_glb_mat)

    return
  end subroutine find_grid_points_with_max_self_density_gpu

  subroutine  compute_scdm_orbital_spreads_in_lattice_dimensions_gpu()
    !! [level 2] compute and wave function linear spreads in units of lattice vector lengths
    implicit none
     real(wp) :: sumx,sumy,sumz
    if (.not.eval_spreads) return
    if (.not.allocated(spr_li)) allocate(spr_li(3,nobtl))
    sum_rho = zero_wp; spr_li = zero_wp
    !$acc parallel loop gang private(ix,iy,iz,rho,ig,ds,sum_rho_,sumx,sumy,sumz)
    do iobtl = 1, nobtl
      sum_rho_ = zero_wp; sumx=zero_wp;sumy=zero_wp;sumz=zero_wp
      !$acc loop vector collapse(3) reduction(+:sumx,sumy,sumz, sum_rho_) private(rho, ig, il, ds, sumx,sumy,sumz, sum_rho_)
      do iz = my_nr(1,3), my_nr(2,3)
        do iy = my_nr(1,2), my_nr(2,2)
          do ix = 1, ngrid(1)
            ig = ix + (iy-1)*ngrid(1) + (iz-1)*ngrid(1)*ngrid(2)
            il = ig - my_ngtot_off
            rho = psi_ks_t(iobtl,il)*psi_ks_t(iobtl,il)
            sum_rho_ = sum_rho_ + rho

            ds(1) = scdm_int2wp(ix-orig_offset)/scdm_int2wp(ngrid(1)) - swfc(1,iobtl)
            ds(2) = scdm_int2wp(iy-orig_offset)/scdm_int2wp(ngrid(2)) - swfc(2,iobtl)
            ds(3) = scdm_int2wp(iz-orig_offset)/scdm_int2wp(ngrid(3)) - swfc(3,iobtl)

            ds(1) = ds(1) - nint(ds(1))
            ds(2) = ds(2) - nint(ds(2))
            ds(3) = ds(3) - nint(ds(3))
              
            sumx = sumx + ds(1)*ds(1) * rho
            sumy = sumy + ds(2)*ds(2) * rho
            sumz = sumz + ds(3)*ds(3) * rho
          end do ! ix
        end do ! iy
      end do ! iz
      !$acc end loop
      sum_rho(iobtl) = sum_rho_
      spr_li(1,iobtl) = sumx
      spr_li(2,iobtl) = sumy
      spr_li(3,iobtl) = sumz
    end do ! iobtl
    !$acc end parallel
#if defined(__MPI)
    !$acc update self(spr_li, sum_rho)
    call MPI_ALLREDUCE(MPI_IN_PLACE, sum_rho, nobtl,                 &
      &                MPI_WP, MPI_SUM, comm_scdm, ierr)
    call MPI_ALLREDUCE(MPI_IN_PLACE, spr_li, 3*nobtl,                &
      &                MPI_WP, MPI_SUM, comm_scdm, ierr)
    !$acc update device(spr_li, sum_rho)
#endif

    !$acc parallel loop
    do iobtl = 1, nobtl
      spr_li(:,iobtl) = sqrt(spr_li(:,iobtl)/sum_rho(iobtl))
    end do
    !$acc end parallel 

    if (is_root.and.mode_debug.ge.DUMP_SPREADS) then
      !$acc update self(spr_li)
      print *, 'scdm_qecp: spreads printed in fort.800'
      write(800,'(3F15.7)') spr_li(:,:)
    end if
    return
  end subroutine compute_scdm_orbital_spreads_in_lattice_dimensions_gpu
#endif

  subroutine  scdm_qecp_optional_features(eval_spreads_)
    !! [level 1] pre-initialization step (optional step 0)
    implicit none
    logical, intent(in), optional :: eval_spreads_
    if (present(eval_spreads_)) eval_spreads = eval_spreads_
    return
  end subroutine scdm_qecp_optional_features

end module scdm_qecp

