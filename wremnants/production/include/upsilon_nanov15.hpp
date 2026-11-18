#ifndef WREMNANTS_UPSILON_NANOV15_HPP
#define WREMNANTS_UPSILON_NANOV15_HPP

#include <algorithm>
#include <array>
#include <cstdlib>
#include <vector>

namespace wrem {

enum class UpsilonAncestorFailure {
  success,
  invalidGenMatch,
  noUpsilonPlus,
  noUpsilonMinus,
  noCommonUpsilon,
  ancestryCycle,
  ancestryDepth,
};

inline int upsilonStateFromPdgId(int pdgId) {
  switch (std::abs(pdgId)) {
  case 553: return 1;
  case 100553: return 2;
  case 200553: return 3;
  default: return 0;
  }
}

template <class MotherVec, class PdgVec>
UpsilonAncestorFailure upsilonAncestorChain(int index, const MotherVec &mothers,
                                           const PdgVec &pdgIds,
                                           std::vector<int> &upsilons) {
  constexpr int maxDepth = 128;
  const int size = static_cast<int>(mothers.size());
  std::array<int, maxDepth> visited;
  int depth = 0;
  while (index >= 0 && index < size) {
    if (depth == maxDepth) return UpsilonAncestorFailure::ancestryDepth;
    const auto end = visited.begin() + depth;
    if (std::find(visited.begin(), end, index) != end)
      return UpsilonAncestorFailure::ancestryCycle;
    visited[depth++] = index;
    if (upsilonStateFromPdgId(pdgIds[index])) upsilons.push_back(index);
    index = mothers[index];
  }
  return UpsilonAncestorFailure::success;
}

// Return {state, failure code}. Once two single-mother paths meet, their
// ancestry is identical; the first shared Upsilon is thus the nearest on both.
template <class MotherVec, class PdgVec>
std::array<int, 2> commonUpsilonAncestor(int plusIndex, int minusIndex,
                                        const MotherVec &mothers,
                                        const PdgVec &pdgIds) {
  using Failure = UpsilonAncestorFailure;
  const auto failure = [](Failure code) {
    return std::array<int, 2>{0, static_cast<int>(code)};
  };
  const int size = static_cast<int>(mothers.size());
  if (mothers.size() != pdgIds.size() || plusIndex < 0 || minusIndex < 0 ||
      plusIndex >= size || minusIndex >= size)
    return failure(Failure::invalidGenMatch);

  std::vector<int> plus, minus;
  auto code = upsilonAncestorChain(plusIndex, mothers, pdgIds, plus);
  if (code != Failure::success) return failure(code);
  code = upsilonAncestorChain(minusIndex, mothers, pdgIds, minus);
  if (code != Failure::success) return failure(code);
  if (plus.empty()) return failure(Failure::noUpsilonPlus);
  if (minus.empty()) return failure(Failure::noUpsilonMinus);

  for (int index : plus) {
    if (std::find(minus.begin(), minus.end(), index) != minus.end())
      return {upsilonStateFromPdgId(pdgIds[index]), 0};
  }
  return failure(Failure::noCommonUpsilon);
}

} // namespace wrem

#endif
