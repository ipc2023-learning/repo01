(define (problem depot-2-1-3-3-3-7) (:domain depots)
(:objects
	depot0 depot1 - Depot
	distributor0 - Distributor
	truck0 truck1 truck2 - Truck
	pallet0 pallet1 pallet2 - Pallet
	crate0 crate1 crate2 crate3 crate4 crate5 crate6 - Crate
	hoist0 hoist1 hoist2 - Hoist)
(:init
	(at pallet0 depot0)
	(clear crate5)
	(at pallet1 depot1)
	(clear crate6)
	(at pallet2 distributor0)
	(clear crate4)
	(at truck0 depot1)
	(at truck1 depot1)
	(at truck2 distributor0)
	(at hoist0 depot0)
	(available hoist0)
	(at hoist1 depot1)
	(available hoist1)
	(at hoist2 distributor0)
	(available hoist2)
	(at crate0 depot0)
	(on crate0 pallet0)
	(at crate1 distributor0)
	(on crate1 pallet2)
	(at crate2 depot1)
	(on crate2 pallet1)
	(at crate3 depot1)
	(on crate3 crate2)
	(at crate4 distributor0)
	(on crate4 crate1)
	(at crate5 depot0)
	(on crate5 crate0)
	(at crate6 depot1)
	(on crate6 crate3)
)

(:goal (and
		(on crate0 pallet0)
		(on crate1 crate5)
		(on crate2 crate4)
		(on crate4 crate1)
		(on crate5 crate0)
		(on crate6 pallet1)
	)
))
