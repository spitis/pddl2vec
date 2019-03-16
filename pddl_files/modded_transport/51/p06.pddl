; Transport city-sequential-30nodes-1000size-4degree-100mindistance-4trucks-12packages-2008seed

(define (problem transport-city-sequential-30nodes-1000size-4degree-100mindistance-4trucks-12packages-2008seed)
 (:domain transport)
 (:objects
  city-loc-1 - location
  city-loc-2 - location
  city-loc-3 - location
  city-loc-4 - location
  city-loc-5 - location
  city-loc-6 - location
  city-loc-7 - location
  city-loc-8 - location
  city-loc-9 - location
  city-loc-10 - location
  city-loc-11 - location
  city-loc-12 - location
  city-loc-13 - location
  city-loc-14 - location
  city-loc-15 - location
  city-loc-16 - location
  city-loc-17 - location
  city-loc-18 - location
  city-loc-19 - location
  city-loc-20 - location
  city-loc-21 - location
  city-loc-22 - location
  city-loc-23 - location
  city-loc-24 - location
  city-loc-25 - location
  city-loc-26 - location
  city-loc-27 - location
  city-loc-28 - location
  city-loc-29 - location
  city-loc-30 - location
  truck-1 - vehicle
  truck-2 - vehicle
  truck-3 - vehicle
  truck-4 - vehicle
  package-1 - package
  package-2 - package
  package-3 - package
  package-4 - package
  package-5 - package
  package-6 - package
  package-7 - package
  package-8 - package
  package-9 - package
  package-10 - package
  package-11 - package
  package-12 - package
  capacity-0 - capacity-number
  capacity-1 - capacity-number
  capacity-2 - capacity-number
  capacity-3 - capacity-number
  capacity-4 - capacity-number
 )
 (:init
  (capacity-predecessor capacity-0 capacity-1)
  (capacity-predecessor capacity-1 capacity-2)
  (capacity-predecessor capacity-2 capacity-3)
  (capacity-predecessor capacity-3 capacity-4)
  (road city-loc-4 city-loc-2)
  (road city-loc-2 city-loc-4)
  (road city-loc-5 city-loc-1)
  (road city-loc-1 city-loc-5)
  (road city-loc-6 city-loc-2)
  (road city-loc-2 city-loc-6)
  (road city-loc-7 city-loc-2)
  (road city-loc-2 city-loc-7)
  (road city-loc-7 city-loc-6)
  (road city-loc-6 city-loc-7)
  (road city-loc-8 city-loc-2)
  (road city-loc-2 city-loc-8)
  (road city-loc-8 city-loc-3)
  (road city-loc-3 city-loc-8)
  (road city-loc-8 city-loc-7)
  (road city-loc-7 city-loc-8)
  (road city-loc-11 city-loc-2)
  (road city-loc-2 city-loc-11)
  (road city-loc-11 city-loc-6)
  (road city-loc-6 city-loc-11)
  (road city-loc-11 city-loc-9)
  (road city-loc-9 city-loc-11)
  (road city-loc-12 city-loc-4)
  (road city-loc-4 city-loc-12)
  (road city-loc-14 city-loc-3)
  (road city-loc-3 city-loc-14)
  (road city-loc-14 city-loc-7)
  (road city-loc-7 city-loc-14)
  (road city-loc-14 city-loc-8)
  (road city-loc-8 city-loc-14)
  (road city-loc-15 city-loc-1)
  (road city-loc-1 city-loc-15)
  (road city-loc-15 city-loc-5)
  (road city-loc-5 city-loc-15)
  (road city-loc-15 city-loc-6)
  (road city-loc-6 city-loc-15)
  (road city-loc-15 city-loc-7)
  (road city-loc-7 city-loc-15)
  (road city-loc-15 city-loc-14)
  (road city-loc-14 city-loc-15)
  (road city-loc-16 city-loc-3)
  (road city-loc-3 city-loc-16)
  (road city-loc-16 city-loc-8)
  (road city-loc-8 city-loc-16)
  (road city-loc-17 city-loc-1)
  (road city-loc-1 city-loc-17)
  (road city-loc-17 city-loc-5)
  (road city-loc-5 city-loc-17)
  (road city-loc-18 city-loc-10)
  (road city-loc-10 city-loc-18)
  (road city-loc-19 city-loc-2)
  (road city-loc-2 city-loc-19)
  (road city-loc-19 city-loc-5)
  (road city-loc-5 city-loc-19)
  (road city-loc-19 city-loc-6)
  (road city-loc-6 city-loc-19)
  (road city-loc-19 city-loc-7)
  (road city-loc-7 city-loc-19)
  (road city-loc-19 city-loc-9)
  (road city-loc-9 city-loc-19)
  (road city-loc-19 city-loc-11)
  (road city-loc-11 city-loc-19)
  (road city-loc-19 city-loc-15)
  (road city-loc-15 city-loc-19)
  (road city-loc-20 city-loc-1)
  (road city-loc-1 city-loc-20)
  (road city-loc-20 city-loc-5)
  (road city-loc-5 city-loc-20)
  (road city-loc-20 city-loc-7)
  (road city-loc-7 city-loc-20)
  (road city-loc-20 city-loc-14)
  (road city-loc-14 city-loc-20)
  (road city-loc-20 city-loc-15)
  (road city-loc-15 city-loc-20)
  (road city-loc-20 city-loc-17)
  (road city-loc-17 city-loc-20)
  (road city-loc-21 city-loc-10)
  (road city-loc-10 city-loc-21)
  (road city-loc-21 city-loc-18)
  (road city-loc-18 city-loc-21)
  (road city-loc-22 city-loc-1)
  (road city-loc-1 city-loc-22)
  (road city-loc-22 city-loc-10)
  (road city-loc-10 city-loc-22)
  (road city-loc-22 city-loc-17)
  (road city-loc-17 city-loc-22)
  (road city-loc-23 city-loc-3)
  (road city-loc-3 city-loc-23)
  (road city-loc-23 city-loc-8)
  (road city-loc-8 city-loc-23)
  (road city-loc-23 city-loc-16)
  (road city-loc-16 city-loc-23)
  (road city-loc-24 city-loc-8)
  (road city-loc-8 city-loc-24)
  (road city-loc-24 city-loc-12)
  (road city-loc-12 city-loc-24)
  (road city-loc-24 city-loc-16)
  (road city-loc-16 city-loc-24)
  (road city-loc-24 city-loc-23)
  (road city-loc-23 city-loc-24)
  (road city-loc-25 city-loc-13)
  (road city-loc-13 city-loc-25)
  (road city-loc-26 city-loc-3)
  (road city-loc-3 city-loc-26)
  (road city-loc-26 city-loc-14)
  (road city-loc-14 city-loc-26)
  (road city-loc-26 city-loc-17)
  (road city-loc-17 city-loc-26)
  (road city-loc-26 city-loc-20)
  (road city-loc-20 city-loc-26)
  (road city-loc-27 city-loc-21)
  (road city-loc-21 city-loc-27)
  (road city-loc-27 city-loc-25)
  (road city-loc-25 city-loc-27)
  (road city-loc-27 city-loc-26)
  (road city-loc-26 city-loc-27)
  (road city-loc-28 city-loc-10)
  (road city-loc-10 city-loc-28)
  (road city-loc-28 city-loc-22)
  (road city-loc-22 city-loc-28)
  (road city-loc-29 city-loc-1)
  (road city-loc-1 city-loc-29)
  (road city-loc-29 city-loc-5)
  (road city-loc-5 city-loc-29)
  (road city-loc-29 city-loc-17)
  (road city-loc-17 city-loc-29)
  (road city-loc-29 city-loc-20)
  (road city-loc-20 city-loc-29)
  (road city-loc-29 city-loc-22)
  (road city-loc-22 city-loc-29)
  (road city-loc-30 city-loc-5)
  (road city-loc-5 city-loc-30)
  (road city-loc-30 city-loc-9)
  (road city-loc-9 city-loc-30)
  (road city-loc-30 city-loc-15)
  (road city-loc-15 city-loc-30)
  (road city-loc-30 city-loc-19)
  (road city-loc-19 city-loc-30)
  (at package-1 city-loc-4)
  (at package-2 city-loc-23)
  (at package-3 city-loc-9)
  (at package-4 city-loc-7)
  (at package-5 city-loc-6)
  (at package-6 city-loc-13)
  (at package-7 city-loc-1)
  (at package-8 city-loc-16)
  (at package-9 city-loc-11)
  (at package-10 city-loc-12)
  (at package-11 city-loc-19)
  (at package-12 city-loc-10)
  (at truck-1 city-loc-25)
  (capacity truck-1 capacity-2)
  (at truck-2 city-loc-29)
  (capacity truck-2 capacity-2)
  (at truck-3 city-loc-29)
  (capacity truck-3 capacity-4)
  (at truck-4 city-loc-28)
  (capacity truck-4 capacity-3)
 )
 (:goal (and
  (at package-1 city-loc-28)
  (at package-2 city-loc-5)
  (at package-3 city-loc-5)
  (at package-4 city-loc-30)
  (at package-5 city-loc-29)
  (at package-6 city-loc-4)
  (at package-7 city-loc-28)
  (at package-8 city-loc-27)
  (at package-9 city-loc-25)
  (at package-10 city-loc-5)
  (at package-11 city-loc-10)
  (at package-12 city-loc-6)
 ))
)
