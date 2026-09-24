use crate::{Result, cuda_graph};
use std::sync::{Barrier, mpsc};

#[test]
fn replay_is_shared_and_capture_excludes_other_native_work() -> Result<()> {
    let barrier = Barrier::new(4);
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    let _replay = cuda_graph::shared().unwrap();
                    // All four readers must be able to hold access simultaneously.
                    barrier.wait();
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
    });
    let capture = cuda_graph::exclusive()?;
    let (started_tx, started_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        started_tx.send(()).unwrap();
        let _replay = cuda_graph::shared().unwrap();
        done_tx.send(()).unwrap();
    });
    started_rx.recv()?;
    assert!(
        done_rx
            .recv_timeout(std::time::Duration::from_millis(30))
            .is_err()
    );
    drop(capture);
    done_rx.recv_timeout(std::time::Duration::from_secs(5))?;
    worker.join().unwrap();
    drop(cuda_graph::teardown());
    Ok(())
}
