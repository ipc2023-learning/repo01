(define (problem depot-1-2-2-3-3-3) (:domain depots)
(:objects
	depot0 - Depot
	distributor0 distributor1 - Distributor
	truck0 truck1 - Truck
	pallet0 pallet1 pallet2 - Pallet
	crate0 crate1 crate2 - Crate
	hoist0 hoist1 hoist2 - Hoist)
(:init
	(at pallet0 depot0)
	(clear crate2)
	(at pallet1 distributor0)
	(clear crate0)
	(at pallet2 distributor1)
	(clear crate1)
	(at truck0 distributor1)
	(at truck1 depot0)
	(at hoist0 depot0)
	(available hoist0)
	(at hoist1 distributor0)
	(available hoist1)
	(at hoist2 distributor1)
	(available hoist2)
	(at crate0 distributor0)
	(on crate0 pallet1)
	(at crate1 distributor1)
	(on crate1 pallet2)
	(at crate2 depot0)
	(on crate2 pallet0)
)

(:goal (and
		(on crate0 crate1)
		(on crate1 pallet1)
	)
))
