! Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
!
! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at
!
!     http://www.apache.org/licenses/LICENSE-2.0
!
! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
! See the License for the specific language governing permissions and
! limitations under the License.

! Test 15.5.2.5 constraints and restrictions for POINTER & ALLOCATABLE
! arguments when both sides of the call have the same attributes.

module m

  type :: t
  end type
  type, extends(t) :: t2
  end type
  type :: pdt(n)
    integer, len :: n
  end type

  type(t), pointer :: mp(:), mpmat(:,:)
  type(t), allocatable :: ma(:), mamat(:,:)
  class(t), pointer :: pp(:)
  class(t), allocatable :: pa(:)
  class(t2), pointer :: pp2(:)
  class(t2), allocatable :: pa2(:)
  class(*), pointer :: up(:)
  class(*), allocatable :: ua(:)
  !ERROR: An assumed (*) type parameter may be used only for a dummy argument, associate name, or named constant
  type(pdt(*)), pointer :: amp(:)
  !ERROR: An assumed (*) type parameter may be used only for a dummy argument, associate name, or named constant
  type(pdt(*)), allocatable :: ama(:)
  type(pdt(:)), pointer :: dmp(:)
  type(pdt(:)), allocatable :: dma(:)
  type(pdt(1)), pointer :: nmp(:)
  type(pdt(1)), allocatable :: nma(:)

 contains

  subroutine smp(x)
    type(t), pointer :: x(:)
  end subroutine
  subroutine sma(x)
    type(t), allocatable :: x(:)
  end subroutine
  subroutine spp(x)
    class(t), pointer :: x(:)
  end subroutine
  subroutine spa(x)
    class(t), allocatable :: x(:)
  end subroutine
  subroutine sup(x)
    class(*), pointer :: x(:)
  end subroutine
  subroutine sua(x)
    class(*), allocatable :: x(:)
  end subroutine
  subroutine samp(x)
    type(pdt(*)), pointer :: x(:)
  end subroutine
  subroutine sama(x)
    type(pdt(*)), allocatable :: x(:)
  end subroutine
  subroutine sdmp(x)
    type(pdt(:)), pointer :: x(:)
  end subroutine
  subroutine sdma(x)
    type(pdt(:)), allocatable :: x(:)
  end subroutine
  subroutine snmp(x)
    type(pdt(1)), pointer :: x(:)
  end subroutine
  subroutine snma(x)
    type(pdt(1)), allocatable :: x(:)
  end subroutine

  subroutine test
    call smp(mp) ! ok
    call sma(ma) ! ok
    call spp(pp) ! ok
    call spa(pa) ! ok
    !ERROR: If a POINTER or ALLOCATABLE dummy or actual argument is polymorphic, both must be so
    call smp(pp)
    !ERROR: If a POINTER or ALLOCATABLE dummy or actual argument is polymorphic, both must be so
    call sma(pa)
    !ERROR: If a POINTER or ALLOCATABLE dummy or actual argument is polymorphic, both must be so
    call spp(mp)
    !ERROR: If a POINTER or ALLOCATABLE dummy or actual argument is polymorphic, both must be so
    call spa(ma)
    !ERROR: If a POINTER or ALLOCATABLE dummy or actual argument is unlimited polymorphic, both must be so
    call sup(pp)
    !ERROR: If a POINTER or ALLOCATABLE dummy or actual argument is unlimited polymorphic, both must be so
    call sua(pa)
    !ERROR: actual argument type 'CLASS(*)' is not compatible with dummy argument type 't'
    call spp(up)
    !ERROR: actual argument type 'CLASS(*)' is not compatible with dummy argument type 't'
    call spa(ua)
    !ERROR: POINTER or ALLOCATABLE dummy and actual arguments must have the same declared type
    call spp(pp2)
    !ERROR: POINTER or ALLOCATABLE dummy and actual arguments must have the same declared type
    call spa(pa2)
    !ERROR: Rank of dummy argument is 1, but actual argument has rank 2
    call smp(mpmat)
    !ERROR: Rank of dummy argument is 1, but actual argument has rank 2
    call sma(mamat)
    call sdmp(dmp) ! ok
    call sdma(dma) ! ok
    call snmp(nmp) ! ok
    call snma(nma) ! ok
    call samp(nmp) ! ok
    call sama(nma) ! ok
    !ERROR: Dummy and actual arguments must defer the same type parameters when POINTER or ALLOCATABLE
    call sdmp(nmp)
    !ERROR: Dummy and actual arguments must defer the same type parameters when POINTER or ALLOCATABLE
    call sdma(nma)
    !ERROR: Dummy and actual arguments must defer the same type parameters when POINTER or ALLOCATABLE
    call snmp(dmp)
    !ERROR: Dummy and actual arguments must defer the same type parameters when POINTER or ALLOCATABLE
    call snma(dma)
    !ERROR: Dummy and actual arguments must defer the same type parameters when POINTER or ALLOCATABLE
    call samp(dmp)
    !ERROR: Dummy and actual arguments must defer the same type parameters when POINTER or ALLOCATABLE
    call sama(dma)
  end subroutine

end module
