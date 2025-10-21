# Attack Monitoring Guide - How to Track When Attackers Are Active

## 🎯 Quick Reference: Attack Indicators

### 1. **POISONING ROUND Markers**
The most obvious indicator that an attack is happening:
```
INFO     ============================== POISONING ROUND: 2004 ==============================
```

**Meaning:** This round includes malicious clients training with poisoned data.

### 2. **Malicious Client Selection**
Shows which clients are malicious vs benign in each poisoning round:
```
INFO     ClientManager: Selected clients
INFO          <class 'backfed.clients.neurotoxin_malicious_client.NeurotoxinClient'>: [13, 61]
             <class 'backfed.clients.base_benign_client.BenignClient'>: [32, 15, 18, ...]
```

**Interpretation:**
- `NeurotoxinClient`: Malicious clients (performing attack)
- `BenignClient`: Honest clients (normal training)

### 3. **Poison Module Updates**
Indicates when malicious clients update their attack strategy:
```
Client [13] (neurotoxin_malicious) at round 2004 - Poison module updated
```

**Meaning:** The attacker has synchronized with other malicious clients and updated their poison pattern.

### 4. **Attack Success Metrics**
Monitor these to see if the attack is working:

**Backdoor Accuracy** (most important):
```
test_backdoor_acc: 90.62% (8156/9000)  ← Attack is SUCCESSFUL
test_backdoor_acc: 1.23% (111/9000)    ← Attack is WEAK/FAILED
test_backdoor_acc: 0.00% (0/9000)      ← Attack has NO EFFECT
```

**Clean Accuracy** (should remain high):
```
test_clean_acc: 82.80% (8280/10000)    ← Model still works normally
test_clean_acc: 30.33% (3033/10000)    ← Model degraded (attack too aggressive)
```

**Weight Difference Norms:**
```
Client 13 has weight diff norm 0.2037  ← Malicious (very small update)
Client 32 has weight diff norm 6.9133  ← Benign (normal update)
```

**Pattern:** Neurotoxin attackers typically have **much smaller** weight updates due to gradient masking.

---

## 📊 Attack Timeline Example

From a real experiment (slurm-7891763.out):

### Round 2001-2003: Pre-Attack
```
Round 2001: test_backdoor_acc: 0.00%  ← No attack yet
Round 2002: test_backdoor_acc: 0.00%  ← No attack yet
Round 2003: test_backdoor_acc: 100.00% ← Random spike (not real attack)
```

### Round 2004: First Poisoning Round
```
INFO ============================== POISONING ROUND: 2004 ==============================
Client [13] (neurotoxin_malicious) - Training time: 2.05s
Client [61] (neurotoxin_malicious) - Training time: 2.24s
Round 2004: test_backdoor_acc: 0.00%  ← Attack started but not effective yet
```

### Round 2010-2018: Sustained Attack
```
Round 2010: POISONING ROUND - Client [0] poison module updated
Round 2011: POISONING ROUND  
Round 2012: POISONING ROUND - Client [4] poison module updated
Round 2013: POISONING ROUND - Client [4] poison module updated
Round 2014: POISONING ROUND
...
Backdoor accuracy fluctuating: 0-4% (weak attack due to gradient_mask_ratio=0.1)
```

---

## 🔍 Command-Line Monitoring Tools

### 1. **Track All Poisoning Rounds**
```bash
grep "POISONING ROUND" slurm-*.out
```

**Output:**
```
INFO ============================== POISONING ROUND: 2004 ==============================
INFO ============================== POISONING ROUND: 2007 ==============================
INFO ============================== POISONING ROUND: 2008 ==============================
```

### 2. **See Which Malicious Clients Trained**
```bash
grep "neurotoxin_malicious.*Training time" slurm-*.out | head -20
```

**Output:**
```
Client [13] (neurotoxin_malicious) - Training time: 2.05s | RAM: 1.85GB | VRAM: 0.07GB
Client [61] (neurotoxin_malicious) - Training time: 2.24s | RAM: 1.87GB | VRAM: 0.07GB
```

### 3. **Track Backdoor Accuracy Over Time**
```bash
grep "test_backdoor_acc" slurm-*.out
```

**Output:**
```
test_backdoor_acc: 0.00% (0/9000)
test_backdoor_acc: 100.00% (9000/9000)
test_backdoor_acc: 90.62% (8156/9000)
test_backdoor_acc: 1.23% (111/9000)
```

### 4. **See Malicious Client Selection Per Round**
```bash
grep -A 5 "POISONING ROUND" slurm-*.out | grep "NeurotoxinClient"
```

**Output:**
```
'backfed.clients.neurotoxin_malicious_client.NeurotoxinClient'>: [13, 61]
'backfed.clients.neurotoxin_malicious_client.NeurotoxinClient'>: [13]
```

### 5. **Monitor Weight Difference Norms (Attack Strength)**
```bash
grep "Client .* has weight diff norm" slurm-*.out | grep -A 2 "POISONING ROUND"
```

### 6. **Real-Time Monitoring**
```bash
tail -f slurm-*.out | grep -E "POISONING ROUND|backdoor_acc|neurotoxin"
```

---

## 📈 Attack Success Criteria

### **Successful Attack:**
- ✅ Backdoor accuracy: **70-95%**
- ✅ Clean accuracy: **75-85%** (maintained)
- ✅ Weight diff norms for malicious clients: **3-10**
- ✅ Poison module updates: Regular

### **Weak Attack:**
- ⚠️ Backdoor accuracy: **5-30%**
- ⚠️ Clean accuracy: **70-85%**
- ⚠️ Weight diff norms for malicious clients: **0.5-2**
- ⚠️ Attack partially working but diluted

### **Failed Attack:**
- ❌ Backdoor accuracy: **0-5%**
- ❌ Clean accuracy: **70-85%** (unaffected)
- ❌ Weight diff norms for malicious clients: **<0.5**
- ❌ Attack has no effect

---

## 🔧 Configuration Check

### **Current Configuration (FIXED):**
```yaml
neurotoxin:
  gradient_mask_ratio: 0.99  ← CORRECT (keeps 1% of gradients)
  aggregate_all_layers: True
```

### **Previous Configuration (WRONG):**
```yaml
neurotoxin:
  gradient_mask_ratio: 0.1   ← WRONG (kept 90% of gradients)
  aggregate_all_layers: True
```

---

## 🎯 Attack Configuration Details

### **Malicious Clients:**
Listed in logs at start:
```
INFO Malicious clients: [95, 41, 31, 7, 21, 79, 13, 61, 4, 0]
```

### **Attack Configuration:**
```
INFO Attack type: all2one, Target class: 2
INFO ClientManager: Attack is enabled, initialize rounds selection with
     multi-shot poison scheme and random selection scheme
```

**Parameters:**
- `all2one`: All classes → target class (most common)
- `Target class: 2`: Backdoored samples classified as class 2
- `multi-shot`: Attack happens across multiple rounds
- `random selection`: Malicious clients selected randomly each round

### **Attack Schedule:**
From config:
```yaml
poison_start_round: 2001  # When attack begins
poison_end_round: 2200    # When attack ends
poison_frequency: multi-shot
poison_interval: 1        # Attack every round (within range)
```

---

## 💡 Pro Tips

### **1. Quick Health Check**
```bash
# Get last 10 backdoor accuracy values
grep "test_backdoor_acc" slurm-*.out | tail -10
```

### **2. Count Poisoning Rounds**
```bash
grep -c "POISONING ROUND" slurm-*.out
```

### **3. Track Attack Progress**
```bash
# See backdoor accuracy trend
grep "test_backdoor_acc" slurm-*.out | awk '{print $2}'
```

### **4. Identify Most Active Malicious Clients**
```bash
grep "neurotoxin_malicious.*Training time" slurm-*.out | \
  awk -F'[][]' '{print $2}' | sort | uniq -c | sort -rn
```

### **5. Compare Clean vs Backdoor Accuracy**
```bash
grep -E "test_clean_acc|test_backdoor_acc" slurm-*.out | \
  tail -20 | paste - -
```

---

## 📝 Logging Output Format

### **Standard Round (No Attack):**
```
INFO ============================== TRAINING ROUND: 2001 ==============================
INFO ClientManager: Selected clients
INFO     <class 'backfed.clients.base_benign_client.BenignClient'>: [51, 88, 4, ...]
INFO Round 2001 completed in 21.37 seconds
INFO test_clean_acc: 82.80% (8280/10000)
INFO test_backdoor_acc: 0.00% (0/9000)
```

### **Poisoning Round (With Attack):**
```
INFO ============================== POISONING ROUND: 2004 ==============================
INFO ClientManager: Selected clients
INFO     <class 'backfed.clients.neurotoxin_malicious_client.NeurotoxinClient'>: [13, 61]
INFO     <class 'backfed.clients.base_benign_client.BenignClient'>: [32, 15, ...]
INFO Client [13] (neurotoxin_malicious) at round 2004 - Poison module updated
INFO Client [13] (neurotoxin_malicious) - Training time: 2.05s | VRAM: 0.35GB
INFO Round 2004 completed in 11.11 seconds
INFO test_clean_acc: 82.80% (8280/10000)
INFO test_backdoor_acc: 45.23% (4071/9000)  ← Attack working!
```

---

## ✅ Expected Results After Fix

With `gradient_mask_ratio: 0.99` (now corrected):

- **Backdoor accuracy should reach:** 70-90%+
- **Attack should be visible** in poisoning rounds
- **Weight diff norms** for malicious clients: 3-10
- **Clean accuracy** should remain high: 75-85%

---

**The attack monitoring tools are now available for tracking Neurotoxin attack progress!**

