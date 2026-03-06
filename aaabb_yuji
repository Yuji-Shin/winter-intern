# ======================
# main
# ======================
def main(): # Main entry point of the program.
    parser = argparse.ArgumentParser() # Create a command-line argument parser.
    parser.add_argument("--host", default="192.168.10.11", help="Robot/URSim controller IP") # IP address of the UR robot or URSim controller.
    parser.add_argument("--use-urcap", action="store_true", help="Use External Control URCap flag if available") # Whether to use the External Control URCap flag.
    parser.add_argument("--dry-run", action="store_true", help="Do not execute moveL, only print target pose") # If enabled, motion targets are printed but not executed.

    parser.add_argument("--list-devices", action="store_true", help="List audio input devices and exit") # If enabled, print microphone device list and exit.
    parser.add_argument("--device", type=int, default=None, help="sounddevice input device index (None=default)") # Audio input device index. None means system default.

    parser.add_argument("--sr", type=int, default=16000, help="Sample rate") # Audio sample rate in Hz.
    parser.add_argument("--model", default="base.en", help="Whisper model (tiny.en/base.en/...)") # Whisper model name.
    parser.add_argument("--lang", default="en", help="Whisper language") # Language passed to Whisper.

    parser.add_argument("--chunk-ms", type=int, default=30, help="20~50ms recommended") # Audio callback chunk size in milliseconds.
    parser.add_argument("--energy", type=float, default=0.012, help="RMS threshold") # Energy threshold used for simple voice activity detection.
    parser.add_argument("--min-speech", type=float, default=0.25) # Minimum utterance duration in seconds.
    parser.add_argument("--end-silence", type=float, default=0.55) # Silence duration required to end an utterance.
    parser.add_argument("--max-utt", type=float, default=4.5) # Maximum utterance duration in seconds.
    parser.add_argument("--debug-energy", action="store_true") # If enabled, print RMS energy values continuously.

    parser.add_argument("--intent-bank", default="intent_bank_en.json") # Path to the intent bank JSON file.
    parser.add_argument("--threshold", type=float, default=0.55) # MiniLM router threshold.

    parser.add_argument("--speed", type=float, default=0.05, help="moveL speed (m/s) - start low for real robot") # Base linear motion speed.
    parser.add_argument("--acc", type=float, default=0.10, help="moveL acc (m/s^2) - start low for real robot") # Base linear motion acceleration.
    parser.add_argument("--pain-retract", type=float, default=0.03, help="PAIN retract Z (m)") # Upward retract distance used during PAIN handling.

    parser.add_argument("--default-step", type=float, default=0.01, help="Default step if distance not spoken (m)") # Default movement distance if the user says a direction without a number.
    parser.add_argument("--max-step", type=float, default=0.02, help="Clamp max step per move (m)") # Maximum allowed movement distance per command.
    parser.add_argument("--require-move-verb", action="store_true", help="Require move/go/... verb for rule move parsing") # If enabled, a command must include a move verb to be parsed as a movement command.

    parser.add_argument("--confirm-timeout", type=float, default=15, help="Seconds to wait for yes/no after parsing a move") # Time limit for motion confirmation.

    # XYZ limits
    parser.add_argument("--x-min", type=float, default=-1.0) # Minimum safe x position.
    parser.add_argument("--x-max", type=float, default=+1.0) # Maximum safe x position.
    parser.add_argument("--y-min", type=float, default=-1.0) # Minimum safe y position.
    parser.add_argument("--y-max", type=float, default=+1.0) # Maximum safe y position.
    parser.add_argument("--z-min", type=float, default=0.0) # Minimum safe z position.
    parser.add_argument("--z-max", type=float, default=1.2) # Maximum safe z position.

    args = parser.parse_args() # Parse all command-line arguments into the args object.

    if args.list_devices:
        # If the user only wants to inspect audio devices, do that and exit.
        list_audio_devices()
        return

    xyz_limits = {
        "x": (args.x_min, args.x_max),
        "y": (args.y_min, args.y_max),
        "z": (args.z_min, args.z_max),
    }
    # Store workspace limits in a dictionary for easier access.

    metrics = None # Placeholder for the CSV logger.
    listener = None # Placeholder for the RealtimeWhisper listener.
    rtde_c = None # Placeholder for the RTDE control interface.
    rtde_r = None # Placeholder for the RTDE receive interface.

    # Briefing + ASR mute gate
    tts_gate = threading.Event() # Event used to mute ASR while TTS is speaking.
    briefing = None # Placeholder for the BriefingSystem instance.

    def announce(text: str):
        """
        Safe wrapper around BriefingSystem.announce().

        Behavior:
        - Mute ASR right before TTS starts
        - Automatically unmute after speaking ends
        """
        nonlocal briefing, tts_gate
        # Use the briefing and tts_gate variables from the outer main() scope.
        if briefing is None:
            # If BriefingSystem is unavailable, just print the message.
            print(f"[Briefing (None)] {text}")
            return

        if getattr(briefing, "tts", None) is None:
            # If the object exists but has no TTS engine, still call announce().
            briefing.announce(text)
            return

        if getattr(briefing, "is_speaking", False):
            # Prevent overlapping TTS announcements.
            return

        tts_gate.set() # Mute ASR before speaking.
        briefing.announce(text) # Start the announcement.

        def _watch_and_unmute(): # Helper thread that waits until TTS is done and then unmutes ASR.
            t0 = time.time() # Save current time for a short startup wait window.
            # Wait briefly until speaking actually begins.
            while time.time() - t0 < 0.5 and not getattr(briefing, "is_speaking", False):
                time.sleep(0.01)

            # Keep waiting while the TTS system is speaking.
            while getattr(briefing, "is_speaking", False):
                time.sleep(0.05)

            # Unmute ASR after TTS finishes.
            tts_gate.clear()

        # Run the watcher thread in the background.
        threading.Thread(target=_watch_and_unmute, daemon=True).start()

    try:
        # Main runtime block begins here.
        try:
            # Try to initialize the TTS / briefing system.
            briefing = BriefingSystem()
        except Exception as e:
            # If TTS initialization fails, continue without it.
            briefing = None
            print("[Briefing INIT ERROR]", repr(e))

        metrics_path = os.path.join("logs", f"metrics_{datetime.now():%Y%m%d_%H%M%S}.csv") # Create a timestamped metrics CSV file path.
        metrics = MetricsCSV(metrics_path) # Initialize the CSV logger.
        metrics.log(kind="startup", step="start", duration_ms=0, note=f"csv={metrics_path}") # Log program startup.

        # Print the current configuration for debugging and reproducibility.
        print("=== CONFIG ===")
        print("HOST:", args.host, "| use_urcap:", args.use_urcap)
        print("DRY_RUN:", args.dry_run)
        print("Audio device:", args.device, "| sr:", args.sr)
        print("Whisper model:", args.model, "| lang:", args.lang)
        print("chunk_ms:", args.chunk_ms, "| energy:", args.energy, "| debug_energy:", args.debug_energy)
        print("speed/acc:", args.speed, args.acc, "| pain_retract:", args.pain_retract)
        print("default_step/max_step:", args.default_step, args.max_step, "| require_move_verb:", args.require_move_verb)
        print("confirm_timeout:", args.confirm_timeout)
        print("XYZ limits:", xyz_limits)
        print("intent bank:", args.intent_bank, "| threshold:", args.threshold)
        print("=============\n")

        # Check whether robot network ports are reachable.
        print("[NET] Checking ports ...")
        dash_ok = tcp_connectable(args.host, 29999, timeout_s=1.5) # Check Dashboard port.
        rtde_ok = tcp_connectable(args.host, 30004, timeout_s=1.5) # Check RTDE port.
        # Print port check results.
        print(f"[NET] {args.host}:29999 dashboard -> {'OK' if dash_ok else 'FAIL'}")
        print(f"[NET] {args.host}:30004 RTDE      -> {'OK' if rtde_ok else 'FAIL'}\n")
        # Save port check result to the CSV log.
        metrics.log(kind="startup", step="net_check", duration_ms=0, note=f"29999={dash_ok},30004={rtde_ok}")

        # Load the Whisper ASR model.
        print("[ASR] Loading Whisper:", args.model)
        # Start timing Whisper loading.
        t0 = perf_counter()
        # Load the selected Whisper model.
        asr = whisper.load_model(args.model)
        # Log Whisper loading time.
        metrics.log(kind="startup", step="load_whisper", duration_ms=(perf_counter() - t0) * 1000.0, note=args.model)

        # Load the intent bank and initialize intent router.
        print("[NLU] Loading intent bank + MiniLM retriever/router...")
        t0 = perf_counter() # Measure time to load the intent bank.
        intent_bank = load_intent_bank(args.intent_bank) # Read the intent bank JSON.
        metrics.log(kind="startup", step="load_intent_bank", duration_ms=(perf_counter() - t0) * 1000.0, note=args.intent_bank) # Log intent bank loading time.

        t0 = perf_counter() # Measure router initialization time.
        retriever = MiniLMRetriever(intent_bank) # Create MiniLM retriever.
        router = IntentRouter(retriever, threshold=args.threshold) # Create the intent router using the retriever and threshold.
        # Log router initialization time.
        metrics.log(
            kind="startup",
            step="init_minilm_router",
            duration_ms=(perf_counter() - t0) * 1000.0,
            note=f"threshold={args.threshold}",
        )

        # Connect to the UR RTDE interfaces.
        print("[RTDE] Connecting to:", args.host)
        t0 = perf_counter() # Start timing the RTDE connection.
        rtde_c = make_rtde_control(args.host, use_urcap=args.use_urcap) # Create control interface.
        rtde_r = rtde_receive.RTDEReceiveInterface(args.host) # Create receive interface.
        metrics.log(kind="startup", step="connect_rtde", duration_ms=(perf_counter() - t0) * 1000.0, note=args.host) # Log RTDE connection time.

        state = {
            "armed": False, # Robot is initially not armed, so motion is blocked.
            "speed_scale": 1.0, # Default movement speed multiplier.
            "pending_move": None, # Stores a parsed move waiting for yes/no confirmation. Format: (dx,dy,dz,dir,dist_m,orig_text)
            "pending_t0": 0.0, # Timestamp when pending_move was created.
        }
        # Print user guidance.
        print(
            "\nRealtime listening... (Ctrl+C to quit)\n"
            "- Say 'start' to arm\n"
            "- Say 'stop' to stop+disarm\n"
            "- Say 'disarm' to disarm\n"
            "- Say 'it hurts' to trigger PAIN behavior\n"
            "- Try: 'move up 5 mm', 'move forward 1 cm'\n"
            "- RULE_MOVE는 파싱 후 'yes/confirm'을 말해야 실행됩니다.\n"
        )

        # Speak initial readiness message.
        announce("System ready. Say start to arm.")

        # Create the real-time ASR listener.
        listener = RealtimeWhisper(
            asr,
            sample_rate=args.sr,
            input_device=args.device,
            chunk_ms=args.chunk_ms,
            energy_threshold=args.energy,
            min_speech_sec=args.min_speech,
            end_silence_sec=args.end_silence,
            max_utt_sec=args.max_utt,
            debug_energy=args.debug_energy,
            language=args.lang,
            mute_event=tts_gate,
        )
        listener.start() # Start microphone listening.

        utt_id = 0 # Initialize utterance counter.

        # Main utterance-processing loop.
        for text in listener.listen_texts():
            utt_id += 1 # Increase utterance ID for each recognized utterance.
            asr_ms = float(getattr(listener, "last_asr_ms", 0.0) or 0.0) # Read the last measured ASR runtime.

            print("[TEXT]", text if text else "(empty)") # Print the recognized text.
            if not text:
                # Log empty recognition result and skip further processing.
                metrics.log(kind="utterance", step="empty_text", utt_id=utt_id, duration_ms=0, asr_ms=asr_ms, text="")
                continue

            tl = text.lower() # Lowercase copy of the recognized text.

            # 1) Safety keyword override has the highest priority.
            t0 = perf_counter()
            ov = override_safety(text) # Check whether the text directly matches a safety override.
            ov_ms = (perf_counter() - t0) * 1000.0 # Measure override check time.
            if ov:
                # If a safety override was detected, handle it immediately.
                t1 = perf_counter()
                handle_intent(
                    ov,
                    rtde_c,
                    rtde_r,
                    state,
                    base_speed=args.speed,
                    base_acc=args.acc,
                    pain_retract_z_m=args.pain_retract,
                    xyz_limits=xyz_limits,
                    dry_run=args.dry_run,
                    announce=announce,
                )
                # Measure how long the action handling took.
                act_ms = (perf_counter() - t1) * 1000.0

                # Log the override action.
                metrics.log(
                    kind="utterance",
                    step="override",
                    utt_id=utt_id,
                    duration_ms=f"{ov_ms:.3f}",
                    asr_ms=f"{asr_ms:.3f}",
                    text=text,
                    override_intent=ov,
                    chosen_intent=ov,
                    note=f"action_ms={act_ms:.3f}, armed={state['armed']}, speed_scale={state['speed_scale']}",
                )
                # Print current system state.
                print("[STATE]", state, "\n")
                # Go to next utterance.
                continue

            # 2) Handle confirmation if a move is already pending.
            if state.get("pending_move") is not None:
                # Cancel the pending move if the confirmation timeout has expired.
                if (perf_counter() - state["pending_t0"]) > float(args.confirm_timeout):
                    print("[CONFIRM] timeout -> cancel pending move\n")
                    announce("Confirmation timeout. Cancelled.")
                    metrics.log(
                        kind="utterance",
                        step="pending_timeout_cancel",
                        utt_id=utt_id,
                        duration_ms=0,
                        asr_ms=f"{asr_ms:.3f}",
                        text=text,
                        chosen_intent="PENDING_CANCEL",
                        note="timeout",
                    )
                    state["pending_move"] = None
                    continue

                # If the user said a cancel word, cancel the pending move.
                if any(k in tl for k in CONFIRM_NO):
                    print("[CONFIRM] canceled\n")
                    announce("Cancelled.")
                    metrics.log(
                        kind="utterance",
                        step="pending_cancel",
                        utt_id=utt_id,
                        duration_ms=0,
                        asr_ms=f"{asr_ms:.3f}",
                        text=text,
                        chosen_intent="PENDING_CANCEL",
                        note="user_said_no",
                    )
                    state["pending_move"] = None
                    continue

                # If the user said a confirmation word, execute the pending move.
                if any(k in tl for k in CONFIRM_YES):
                    if not state["armed"]:
                        # Even confirmed moves must be blocked if the system is not armed.
                        print("[BLOCKED] Not armed. Say 'start' first.\n")
                        announce("Not armed. Say start first.")
                        metrics.log(
                            kind="utterance",
                            step="pending_confirm_blocked_not_armed",
                            utt_id=utt_id,
                            duration_ms=0,
                            asr_ms=f"{asr_ms:.3f}",
                            text=text,
                            chosen_intent="PENDING_CONFIRM",
                            note="blocked:not_armed",
                        )
                        state["pending_move"] = None
                        continue

                    dx, dy, dz, found_dir, dist_m, orig_text = state["pending_move"] # Unpack the pending movement information.
                    state["pending_move"] = None # Clear the pending move now that it is being processed.

                    speed = args.speed * state["speed_scale"] # Apply the current speed scaling factor.
                    acc = args.acc * state["speed_scale"] # Apply the current acceleration scaling factor.

                    print(f"[CONFIRMED] executing move: dir={found_dir} dist={dist_m:.3f}m") # Print the move about to be executed.
                    announce("Confirmed. Executing move.") # Speak confirmation feedback.
                    t1 = perf_counter() # Start timing the motion call.
                    ok = moveL_delta(rtde_c, rtde_r, dx, dy, dz, speed, acc, xyz_limits, args.dry_run) # Execute the motion.
                    move_ms = (perf_counter() - t1) * 1000.0 # Measure motion call time.

                    # Log the confirmed rule-based motion.
                    metrics.log(
                        kind="utterance",
                        step="rule_move_confirmed",
                        utt_id=utt_id,
                        duration_ms=f"{move_ms:.3f}",
                        asr_ms=f"{asr_ms:.3f}",
                        text=orig_text,
                        chosen_intent="RULE_MOVE",
                        note=f"confirmed_by={text} ok={ok} dx={dx:.3f} dy={dy:.3f} dz={dz:.3f} speed={speed:.3f}",
                    )
                    # Print motion result.
                    print("moveL ok:", ok, "\n")
                    continue

                # If the utterance was neither yes nor no, keep waiting.
                print("[CONFIRM] Waiting for 'yes/confirm' or 'no/cancel'...\n")
                metrics.log(
                    kind="utterance",
                    step="pending_wait",
                    utt_id=utt_id,
                    duration_ms=0,
                    asr_ms=f"{asr_ms:.3f}",
                    text=text,
                    chosen_intent="PENDING_WAIT",
                    note="not_yes_no",
                )
                continue

            # 3) Parse rule-based movement commands before using MiniLM routing.
            t0 = perf_counter()
            parsed = parse_move_command(
                text,
                default_step_m=float(args.default_step),
                max_step_m=float(args.max_step),
                require_move_verb=bool(args.require_move_verb),
            )
            # Measure parsing time.
            parse_ms = (perf_counter() - t0) * 1000.0

            if parsed:
                # If rule-based move parsing succeeded, handle it before general intent routing.
                if not state["armed"]: # Block motion if the system is not armed.
                    metrics.log(
                        kind="utterance",
                        step="rule_move_blocked_not_armed",
                        utt_id=utt_id,
                        duration_ms=f"{parse_ms:.3f}",
                        asr_ms=f"{asr_ms:.3f}",
                        text=text,
                        chosen_intent="RULE_MOVE",
                        note="blocked:not_armed",
                    )
                    print("[BLOCKED] Not armed. Say 'start' first.\n")
                    announce("Not armed. Say start first.")
                    continue

                dx, dy, dz, found_dir, dist_m = parsed # Unpack parsed movement values.
                state["pending_move"] = (dx, dy, dz, found_dir, dist_m, text) # Save the movement as pending instead of executing immediately.
                state["pending_t0"] = perf_counter() # Save current time for confirmation timeout tracking.

                # Log that a movement was parsed and is waiting for confirmation.
                metrics.log(
                    kind="utterance",
                    step="rule_move_parsed_pending",
                    utt_id=utt_id,
                    duration_ms=f"{parse_ms:.3f}",
                    asr_ms=f"{asr_ms:.3f}",
                    text=text,
                    chosen_intent="RULE_MOVE_PENDING",
                    note=f"dir={found_dir} dist={dist_m:.3f} dx={dx:.3f} dy={dy:.3f} dz={dz:.3f}",
                )
               
                print(f"[RULE_MOVE] Parsed dir={found_dir} dist={dist_m:.3f}m -> dx={dx:.3f} dy={dy:.3f} dz={dz:.3f}") # Print parsed move values.
                print("[CONFIRM] Say 'yes/confirm' to execute, or 'no/cancel' to abort.\n") # Ask the user for confirmation.
                announce("Move parsed. Say confirm to execute, or cancel.") # Speak confirmation prompt.
                continue

            # 4) If not a rule-based move, use MiniLM intent routing.
            t0 = perf_counter() 
            chosen, candidates = router.route(text) # Route the utterance to the most likely intent.
            route_ms = (perf_counter() - t0) * 1000.0 # Measure routing time.
            top3 = top3_str(candidates) # Convert top-3 candidates to a string for logging.

            print("\n[Top-3 candidates]")  # Print top-3 intent candidates.
            for c in candidates[:3]:
                print(f"- {c['intent']:10s} score={c['score']:.3f} ({c['text']})")

            if not chosen:
                # If all candidates are below threshold, do nothing.
                metrics.log(
                    kind="utterance",
                    step="route_no_intent",
                    utt_id=utt_id,
                    duration_ms=f"{route_ms:.3f}",
                    asr_ms=f"{asr_ms:.3f}",
                    text=text,
                    top3=top3,
                    note="below_threshold",
                )
                print("\n[NO_INTENT] below threshold\n")
                continue

            intent = chosen["intent"] # Get final chosen intent.
            score = chosen.get("score", "") # Get its score if present.

            print("\n[CHOSEN]", chosen) # Print the chosen intent dictionary.

            # Safety intents should always be allowed, even when not armed.
            if intent in ["PAIN", "STOP", "START", "DISARM"]:
                t1 = perf_counter()
                handle_intent(
                    intent,
                    rtde_c,
                    rtde_r,
                    state,
                    base_speed=args.speed,
                    base_acc=args.acc,
                    pain_retract_z_m=args.pain_retract,
                    xyz_limits=xyz_limits,
                    dry_run=args.dry_run,
                    announce=announce,
                )
                act_ms = (perf_counter() - t1) * 1000.0 # Measure action handling time.

                # Log safety intent handling.
                metrics.log(
                    kind="utterance",
                    step="route_intent_safety",
                    utt_id=utt_id,
                    duration_ms=f"{route_ms:.3f}",
                    asr_ms=f"{asr_ms:.3f}",
                    text=text,
                    chosen_intent=intent,
                    chosen_score=score,
                    top3=top3,
                    note=f"action_ms={act_ms:.3f}, armed={state['armed']}, speed_scale={state['speed_scale']}",
                )

                print("[STATE]", state, "\n")
                continue

            # Block all non-safety intents if the system is not armed.
            if not state["armed"]:
                metrics.log(
                    kind="utterance",
                    step="route_intent_blocked_not_armed",
                    utt_id=utt_id,
                    duration_ms=f"{route_ms:.3f}",
                    asr_ms=f"{asr_ms:.3f}",
                    text=text,
                    chosen_intent=intent,
                    chosen_score=score,
                    top3=top3,
                    note="blocked:not_armed",
                )
                print("[BLOCKED] Not armed. Say 'start' first.\n")
                announce("Not armed. Say start first.")
                continue

            # Handle the non-safety intent.
            t1 = perf_counter()
            handle_intent(
                intent,
                rtde_c,
                rtde_r,
                state,
                base_speed=args.speed,
                base_acc=args.acc,
                pain_retract_z_m=args.pain_retract,
                xyz_limits=xyz_limits,
                dry_run=args.dry_run,
                announce=announce,
            )
            act_ms = (perf_counter() - t1) * 1000.0 # Measure handling time.

            # Log intent handling.
            metrics.log(
                kind="utterance",
                step="route_intent",
                utt_id=utt_id,
                duration_ms=f"{route_ms:.3f}",
                asr_ms=f"{asr_ms:.3f}",
                text=text,
                chosen_intent=intent,
                chosen_score=score,
                top3=top3,
                note=f"action_ms={act_ms:.3f}, armed={state['armed']}, speed_scale={state['speed_scale']}",
            )

            print("[STATE]", state, "\n") # Print current state after intent handling.

    except KeyboardInterrupt:
        # Handle Ctrl+C cleanly.
        print("\n[QUIT] Ctrl+C")
        if metrics:
            # Log keyboard interrupt event.
            metrics.log(kind="startup", step="keyboard_interrupt", duration_ms=0)
    except Exception as e:
        # Print unexpected runtime error.
        print("\n[ERROR]", repr(e))
        if metrics:
            # Log exception details.
            metrics.log(kind="startup", step="exception", duration_ms=0, note=repr(e))
        raise # Re-raise the exception so the full traceback is still visible.
    finally:
        # Cleanup block that runs whether the program ends normally or with an error.
        if listener:
            # Stop microphone listening.
            listener.stop()
        if rtde_c:
            try:
                # Stop the RTDE-side script if possible.
                rtde_c.stopScript()

            except Exception:
                pass
        if briefing:
            try:
                # Wait for TTS playback to finish before full shutdown.
                briefing.wait_until_finished()
            except Exception:
                pass
        if metrics:
            metrics.log(kind="startup", step="end", duration_ms=0) # Log normal shutdown.
            metrics.close() # Close the CSV file.
        print("[RTDE] Closed") # Final console message.


if __name__ == "__main__":
    # Run main() only when this file is executed directly.
    main()
