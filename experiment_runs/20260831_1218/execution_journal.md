[2026-08-31T16:35:59.523554+00:00] CUDA OK gpu=NVIDIA RTX 6000 Ada Generation mem=47.99GB torch=2.11.0+cu128
[2026-08-31T16:35:59.524562+00:00] Running unit smoke tests (no full training)
[2026-08-31T16:36:20.547575+00:00] pytest rc=0
[2026-08-31T16:36:20.548573+00:00] === DATASET A upgraded ===
[2026-08-31T16:36:20.551719+00:00] START upgraded__dep_legr__legr_gcn__gcn_undirected__seed_42
[2026-08-31T16:38:01.351694+00:00] END upgraded__dep_legr__legr_gcn__gcn_undirected__seed_42 status=VERIFIED
[2026-08-31T16:38:01.354696+00:00] START upgraded__dep_legr__legr_gcn__gcn_undirected_15__seed_42
[2026-08-31T16:40:43.977915+00:00] END upgraded__dep_legr__legr_gcn__gcn_undirected_15__seed_42 status=VERIFIED
[2026-08-31T16:40:43.981987+00:00] START upgraded__task1_sbert__sbert_ft__ged_030__seed_42
[2026-08-31T16:42:27.783582+00:00] END upgraded__task1_sbert__sbert_ft__ged_030__seed_42 status=VERIFIED
[2026-08-31T16:42:27.787663+00:00] START upgraded__task1_sbert__sbert_ft__ged_0__seed_42
[2026-08-31T16:44:11.135618+00:00] END upgraded__task1_sbert__sbert_ft__ged_0__seed_42 status=VERIFIED
[2026-08-31T16:44:11.137618+00:00] START upgraded__task1_sbert__sbert_ft__tied_weights__seed_42
[2026-08-31T16:46:11.738086+00:00] END upgraded__task1_sbert__sbert_ft__tied_weights__seed_42 status=VERIFIED
[2026-08-31T16:46:11.742100+00:00] START upgraded__task4_dirgnn__dirgnn__directed__seed_42
[2026-08-31T16:47:50.181131+00:00] END upgraded__task4_dirgnn__dirgnn__directed__seed_42 status=VERIFIED
[2026-08-31T16:47:50.185226+00:00] START upgraded__task4_dirgnn__dirgnn__tied_in_out__seed_42
[2026-08-31T16:49:28.300373+00:00] END upgraded__task4_dirgnn__dirgnn__tied_in_out__seed_42 status=VERIFIED
[2026-08-31T16:49:28.304523+00:00] START upgraded__task2_latent__legr__action_type_analysis__seed_42
[2026-08-31T16:49:52.054936+00:00] END upgraded__task2_latent__legr__action_type_analysis__seed_42 status=VERIFIED
[2026-08-31T16:49:52.058936+00:00] START upgraded__task3_atomic__legr_15tool__zero_shot_atomic__seed_42
[2026-08-31T16:50:30.642805+00:00] END upgraded__task3_atomic__legr_15tool__zero_shot_atomic__seed_42 status=VERIFIED
[2026-08-31T16:50:30.647806+00:00] START upgraded__task3_atomic__legr_30tool__zero_shot_atomic__seed_42
[2026-08-31T16:50:40.787186+00:00] END upgraded__task3_atomic__legr_30tool__zero_shot_atomic__seed_42 status=FAILED
[2026-08-31T16:50:40.791185+00:00] Dataset A manager gate pass=True
[2026-08-31T16:50:40.791185+00:00] === DATASET B upgraded_v3 ===
[2026-08-31T16:50:40.793262+00:00] START upgraded_v3__dep_legr__legr_gcn__gcn_undirected__seed_42
[2026-08-31T16:51:51.362070+00:00] END upgraded_v3__dep_legr__legr_gcn__gcn_undirected__seed_42 status=FAILED
[2026-08-31T16:51:51.365070+00:00] START upgraded_v3__dep_legr__legr_gcn__gcn_undirected_15__seed_42
[2026-08-31T16:53:49.468847+00:00] END upgraded_v3__dep_legr__legr_gcn__gcn_undirected_15__seed_42 status=VERIFIED
[2026-08-31T16:53:49.472924+00:00] START upgraded_v3__task1_sbert__sbert_ft__ged_030__seed_42
[2026-08-31T16:56:46.564524+00:00] END upgraded_v3__task1_sbert__sbert_ft__ged_030__seed_42 status=VERIFIED
[2026-08-31T16:56:46.568549+00:00] START upgraded_v3__task1_sbert__sbert_ft__ged_0__seed_42
[2026-08-31T16:59:44.250739+00:00] END upgraded_v3__task1_sbert__sbert_ft__ged_0__seed_42 status=VERIFIED
[2026-08-31T16:59:44.255254+00:00] START upgraded_v3__task1_sbert__sbert_ft__tied_weights__seed_42
[2026-08-31T17:02:45.019957+00:00] END upgraded_v3__task1_sbert__sbert_ft__tied_weights__seed_42 status=VERIFIED
[2026-08-31T17:02:45.024479+00:00] START upgraded_v3__task4_dirgnn__dirgnn__directed__seed_42
[2026-08-31T17:04:46.345442+00:00] END upgraded_v3__task4_dirgnn__dirgnn__directed__seed_42 status=VERIFIED
[2026-08-31T17:04:46.349548+00:00] START upgraded_v3__task4_dirgnn__dirgnn__tied_in_out__seed_42
[2026-08-31T17:06:42.112057+00:00] END upgraded_v3__task4_dirgnn__dirgnn__tied_in_out__seed_42 status=VERIFIED
[2026-08-31T17:06:42.117146+00:00] START upgraded_v3__task2_latent__legr__action_type_analysis__seed_42
[2026-08-31T17:06:42.121202+00:00] END upgraded_v3__task2_latent__legr__action_type_analysis__seed_42 status=FAILED
[2026-08-31T17:06:42.127360+00:00] START upgraded_v3__task3_atomic__legr_15tool__zero_shot_atomic__seed_42
[2026-08-31T17:07:25.401688+00:00] END upgraded_v3__task3_atomic__legr_15tool__zero_shot_atomic__seed_42 status=VERIFIED
[2026-08-31T17:07:25.406695+00:00] START upgraded_v3__task3_atomic__legr_30tool__zero_shot_atomic__seed_42
[2026-08-31T17:07:35.603182+00:00] END upgraded_v3__task3_atomic__legr_30tool__zero_shot_atomic__seed_42 status=FAILED
[2026-08-31T17:07:35.618897+00:00] ALL PHASES COMPLETE
[2026-08-31T17:08:33.990920+00:00] RETRY START upgraded_v3__dep_legr__legr_gcn__gcn_undirected__seed_42
[2026-08-31T17:08:33.991463+00:00] START upgraded_v3__dep_legr__legr_gcn__gcn_undirected__seed_42
[2026-08-31T17:10:38.912830+00:00] END upgraded_v3__dep_legr__legr_gcn__gcn_undirected__seed_42 status=VERIFIED
[2026-08-31T17:10:38.916971+00:00] RETRY START upgraded_v3__task2_latent__legr__action_type_analysis__seed_42
[2026-08-31T17:10:38.917972+00:00] START upgraded_v3__task2_latent__legr__action_type_analysis__seed_42
[2026-08-31T17:10:56.656814+00:00] END upgraded_v3__task2_latent__legr__action_type_analysis__seed_42 status=VERIFIED
[2026-08-31T17:10:56.668438+00:00] RETRY COMPLETE
[2026-08-31T17:12:00.000000+00:00] Dataset B manager gate updated: 30-tool GCN + Task 2 VERIFIED after retry; 30-tool atomic remains NOT_SUPPORTED
[2026-08-31T17:12:30.000000+00:00] checkpoint_manifest.json rebuilt with __tool15__/__tool30__ IDs (15-tool no longer overwrites 30-tool)
[2026-08-31T19:03:30.768099+00:00] RETRY 30TOOL ATOMIC COMPLETE both datasets VERIFIED (routing_15 queries, 30-tool encoder)
[2026-08-31T19:02:30.869531+00:00] RETRY START upgraded__task3_atomic__legr_30tool__zero_shot_atomic__seed_42
[2026-08-31T19:02:30.870951+00:00] START upgraded__task3_atomic__legr_30tool__zero_shot_atomic__seed_42
[2026-08-31T19:02:59.124740+00:00] END upgraded__task3_atomic__legr_30tool__zero_shot_atomic__seed_42 status=VERIFIED
[2026-08-31T19:02:59.128890+00:00] RETRY END upgraded__task3_atomic__legr_30tool__zero_shot_atomic__seed_42 status=VERIFIED
[2026-08-31T19:02:59.131890+00:00] RETRY START upgraded_v3__task3_atomic__legr_30tool__zero_shot_atomic__seed_42
[2026-08-31T19:02:59.132888+00:00] START upgraded_v3__task3_atomic__legr_30tool__zero_shot_atomic__seed_42
[2026-08-31T19:03:30.756735+00:00] END upgraded_v3__task3_atomic__legr_30tool__zero_shot_atomic__seed_42 status=VERIFIED
[2026-08-31T19:03:30.761372+00:00] RETRY END upgraded_v3__task3_atomic__legr_30tool__zero_shot_atomic__seed_42 status=VERIFIED
[2026-08-31T19:03:30.768099+00:00] RETRY 30TOOL ATOMIC COMPLETE
[2026-08-31T19:15:54.540356+00:00] START upgraded__task1_sbert__sbert_ft__ged_0__tool15__seed_42
[2026-08-31T19:15:54.541364+00:00] START upgraded__task1_sbert__sbert_ft__ged_0__tool15__seed_42
[2026-08-31T19:20:32.580243+00:00] START upgraded__task1_sbert__sbert_ft__ged_0__tool15__seed_42
[2026-08-31T19:20:32.582253+00:00] START upgraded__task1_sbert__sbert_ft__ged_0__tool15__seed_42
[2026-08-31T19:25:02.966575+00:00] END upgraded__task1_sbert__sbert_ft__ged_0__tool15__seed_42 status=VERIFIED
[2026-08-31T19:25:02.971153+00:00] START upgraded__task1_sbert__sbert_ft__ged_030__tool15__seed_42
[2026-08-31T19:25:02.973241+00:00] START upgraded__task1_sbert__sbert_ft__ged_030__tool15__seed_42
[2026-08-31T19:29:36.109670+00:00] END upgraded__task1_sbert__sbert_ft__ged_030__tool15__seed_42 status=VERIFIED
[2026-08-31T19:29:36.115794+00:00] START upgraded__task1_sbert__sbert_ft__tied_weights__tool15__seed_42
[2026-08-31T19:29:36.117794+00:00] START upgraded__task1_sbert__sbert_ft__tied_weights__tool15__seed_42
[2026-08-31T19:33:55.134864+00:00] END upgraded__task1_sbert__sbert_ft__tied_weights__tool15__seed_42 status=VERIFIED
[2026-08-31T19:33:55.138864+00:00] START upgraded__task1_sbert__sbert_ft__ged_0__tool30__seed_42
[2026-08-31T19:33:55.140949+00:00] START upgraded__task1_sbert__sbert_ft__ged_0__tool30__seed_42
[2026-08-31T19:35:39.746453+00:00] END upgraded__task1_sbert__sbert_ft__ged_0__tool30__seed_42 status=VERIFIED
[2026-08-31T19:35:39.749453+00:00] START upgraded__task1_sbert__sbert_ft__ged_030__tool30__seed_42
[2026-08-31T19:35:39.751595+00:00] START upgraded__task1_sbert__sbert_ft__ged_030__tool30__seed_42
